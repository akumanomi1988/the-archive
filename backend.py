"""
The Archive — Minimal local system for indexing, searching, viewing, and downloading EPUB with optional local LLM enrichment (LMStudio).

Single-file backend with clear sections: config, models, db, indexer, LLM client, API.

Stack: Python 3.11+, FastAPI, SQLModel (SQLAlchemy + Pydantic), SQLite, ebooklib + BeautifulSoup, hashlib, httpx.

Notes:
- Idempotent indexing based on file sha256.
- Does not store full book text in DB; parses on demand for /open.
- Enrichment stored as JSON plus mirrored columns genre/year for filtering.
- Config loaded from SKALD_CONFIG env var or ./config.json; sensible defaults otherwise.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
import time
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, cast
import threading
from contextlib import contextmanager
import unicodedata
import hashlib
import zipfile
from xml.etree import ElementTree as ET
import mimetypes
import posixpath
import tempfile
import shutil
import base64

import bleach
import httpx
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from ebooklib import epub, ITEM_DOCUMENT
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import Column, String
from sqlalchemy.dialects.sqlite import JSON as SQLITE_JSON
from sqlmodel import Field as SQLField, Session, SQLModel, create_engine, select
from sqlalchemy import Integer
from enrichment import (
    ChainEnricher,
    EnrichmentResult,
    OpenLibraryProvider,
    GoogleBooksProvider,
    G4FProvider,
    LMStudioProvider,
)


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------


class AppConfig(BaseModel):
    library_path: str = Field(default=str(Path.cwd() / "library"))
    db_path: str = Field(default=str(Path.cwd() / "skald.db"))
    lm_enabled: bool = False
    lm_url: str = "http://localhost:1234/v1/chat/completions"
    lm_timeout: float = 300.0  # Aumentado a 90 segundos para modelos lentos
    lm_mock: bool = True
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])
    page_size_default: int = 20
    enrichment_batch_size: int = 10
    # Default language used for enrichment summaries if none is provided by the client/frontend
    default_language: str = "es"
    # Optional model name for OpenAI-style LMStudio endpoint
    lm_model: Optional[str] = "openai/gpt-oss-20"
    # LM rate limiting - Máximo 1 libro siendo enriquecido simultáneamente para evitar timeouts
    lm_max_concurrency: int = 1
    lm_min_interval_ms: int = 10000  # Intervalo más largo entre peticiones

    # Providers configuration
    providers_order: List[str] = Field(default_factory=lambda: [
        "openlibrary", "googlebooks", "g4f", "lmstudio"
    ])
    openlibrary_enabled: bool = True
    openlibrary_timeout: float = 8.0
    googlebooks_enabled: bool = True
    googlebooks_timeout: float = 8.0
    g4f_enabled: bool = False
    g4f_model: Optional[str] = None
    g4f_timeout: float = 30.0

    @field_validator("page_size_default", "enrichment_batch_size", "lm_max_concurrency", "lm_min_interval_ms")
    def _positive(cls, v: int):  # type: ignore[override]
        if v <= 0:
            raise ValueError("must be positive")
        return v


def load_config(path_override: Optional[str | Path] = None) -> AppConfig:
    logger = logging.getLogger("skald")
    cfg_path_env = os.environ.get("SKALD_CONFIG")
    candidate_paths: List[Path] = []
    if path_override:
        candidate_paths.append(Path(path_override))
    if cfg_path_env:
        candidate_paths.append(Path(cfg_path_env))
    candidate_paths.append(Path.cwd() / "config.json")

    for p in candidate_paths:
        try:
            if p.exists():
                logger.info("Loading configuration from: %s", p)
                with p.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                config = AppConfig(**data)
                logger.info("Configuration loaded successfully: lm_enabled=%s, lm_timeout=%s, lm_max_concurrency=%s, lm_min_interval_ms=%s", 
                           config.lm_enabled, config.lm_timeout, config.lm_max_concurrency, config.lm_min_interval_ms)
                return config
        except Exception as e:
            logger.warning("Failed loading config from %s: %s", p, e)
    logger.info("Using default in-memory config; create config.json or set SKALD_CONFIG to customize.")
    return AppConfig()


LOGGER = logging.getLogger("archive")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)

CONFIG = load_config()


# ------------------------------------------------------------
# DB Models & Engine
# ------------------------------------------------------------


class Book(SQLModel, table=True):
    id: Optional[int] = SQLField(default=None, primary_key=True)
    title: str = SQLField(index=True)
    author: str = SQLField(index=True)
    language: Optional[str] = SQLField(default=None, index=True)
    path: str = SQLField(index=True)
    size_bytes: int
    modified_iso: str
    sha256: str = SQLField(sa_column=Column(String, unique=True, index=True))
    # Enrichment JSON payload
    enriched: Optional[Dict[str, Any]] = SQLField(default=None, sa_column=Column(SQLITE_JSON))
    enrichment_status: str = SQLField(default="none", index=True)  # none|ok|failed
    # Mirrored fields for filtering without relying on JSON1 availability
    genre: Optional[str] = SQLField(default=None, index=True)
    year: Optional[int] = SQLField(default=None, index=True)
    created_at: datetime = SQLField(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = SQLField(default_factory=lambda: datetime.now(timezone.utc))


class BookState(SQLModel, table=True):
    id: Optional[int] = SQLField(default=None, primary_key=True)
    book_id: int = SQLField(sa_column=Column(Integer, unique=True))
    # User flags
    favorite_flag: bool = SQLField(default=False, index=True)
    read_flag: bool = SQLField(default=False, index=True)
    pending_flag: bool = SQLField(default=False, index=True)
    # Progress
    last_mode: Optional[str] = SQLField(default=None, index=True)  # 'scroll' | 'paged'
    scroll_top: Optional[int] = SQLField(default=None)
    page: Optional[int] = SQLField(default=None)
    percent: Optional[float] = SQLField(default=None)
    updated_at: datetime = SQLField(default_factory=lambda: datetime.now(timezone.utc), index=True)


DB_PATH: Path = Path(CONFIG.db_path)
ENGINE = None  # type: ignore[assignment]
ENRICH_PROGRESS: Dict[str, Any] = {"running": False}
LM_SEMAPHORE: Optional[threading.Semaphore] = None
LM_LAST_REQUEST: float = 0.0
LM_LOCK = threading.Lock()
COVERS_DIR: Optional[Path] = None
# External cover fetch throttling
_COVER_FETCH_LOCK = threading.Lock()
_COVER_LAST_REQUEST: float = 0.0
_COVER_MIN_INTERVAL_MS: int = 400  # ~2.5 req/s globally


@contextmanager
def lm_rate_limit():
    """Global LM request rate limiter: limits concurrent requests and enforces a minimum
    interval between request starts.

    Controlled by CONFIG.lm_max_concurrency and CONFIG.lm_min_interval_ms.
    """
    global LM_LAST_REQUEST
    sem = LM_SEMAPHORE
    if sem is None:
        # Not initialized yet; no limiting
        yield
        return
    sem.acquire()
    try:
        # Enforce min interval between starts
        with LM_LOCK:
            now = time.monotonic()
            min_interval = (CONFIG.lm_min_interval_ms or 0) / 1000.0
            if min_interval > 0:
                delta = now - LM_LAST_REQUEST
                if delta < min_interval:
                    wait_time = min_interval - delta
                    LOGGER.info("LM rate limiting: waiting %.2fs before next request (min_interval=%.2fs)", 
                               wait_time, min_interval)
                    time.sleep(wait_time)
            LM_LAST_REQUEST = time.monotonic()
        yield
    finally:
        sem.release()


def init_app(config_path: Optional[str] = None) -> None:
    """(Re)initialize config and database engine/tables.

    Safe to call multiple times in-process (used in tests).
    """
    global CONFIG, DB_PATH, ENGINE, LM, LM_SEMAPHORE, LM_LAST_REQUEST, COVERS_DIR, _COVER_LAST_REQUEST
    CONFIG = load_config(config_path)
    DB_PATH = Path(CONFIG.db_path)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    ENGINE = create_engine(f"sqlite:///{DB_PATH.as_posix()}", echo=False)
    SQLModel.metadata.create_all(ENGINE)
    # Covers directory next to DB
    COVERS_DIR = DB_PATH.parent / "covers"
    COVERS_DIR.mkdir(parents=True, exist_ok=True)
    _COVER_LAST_REQUEST = 0.0
    # Init LM rate limiter
    max_concurrency = max(1, int(CONFIG.lm_max_concurrency))
    LM_SEMAPHORE = threading.Semaphore(max_concurrency)
    LM_LAST_REQUEST = 0.0
    LOGGER.info("LM rate limiter initialized: max_concurrency=%d, min_interval_ms=%d", 
               max_concurrency, CONFIG.lm_min_interval_ms)
    # Recreate LM client with possibly new config
    LM = LMClient(
        CONFIG.lm_url,
        timeout=CONFIG.lm_timeout,
        enabled=CONFIG.lm_enabled,
        mock=CONFIG.lm_mock,
        model=CONFIG.lm_model,
    )
    LOGGER.info("LM client initialized: url=%s, timeout=%s, enabled=%s, mock=%s, model=%s", 
               CONFIG.lm_url, CONFIG.lm_timeout, CONFIG.lm_enabled, CONFIG.lm_mock, CONFIG.lm_model)

    # Build enrichment chain
    providers: List[Any] = []
    # Construct instances lazily according to order and flags
    for name in CONFIG.providers_order:
        try:
            if name == "openlibrary" and CONFIG.openlibrary_enabled:
                providers.append(OpenLibraryProvider(timeout=CONFIG.openlibrary_timeout))
            elif name == "googlebooks" and CONFIG.googlebooks_enabled:
                providers.append(GoogleBooksProvider(timeout=CONFIG.googlebooks_timeout))
            elif name == "g4f" and CONFIG.g4f_enabled:
                providers.append(G4FProvider(model=CONFIG.g4f_model or "gpt-4o-mini", timeout=CONFIG.g4f_timeout))
            elif name == "lmstudio" and CONFIG.lm_enabled:
                providers.append(LMStudioProvider(base_url=CONFIG.lm_url, model=CONFIG.lm_model, timeout=CONFIG.lm_timeout))
        except Exception as e:
            LOGGER.warning("Failed to init provider %s: %s", name, e)
    if not providers:
        # Ensure we have at least the LMStudio provider if nothing else is enabled
        if CONFIG.lm_enabled:
            providers.append(LMStudioProvider(base_url=CONFIG.lm_url, model=CONFIG.lm_model, timeout=CONFIG.lm_timeout))
    global ENRICH_CHAIN
    ENRICH_CHAIN = ChainEnricher(providers)
    LOGGER.info("Enrichment chain initialized with providers: %s", [getattr(p, 'name', str(p)) for p in providers])


def get_session() -> Iterable[Session]:
    assert ENGINE is not None, "ENGINE not initialized; call init_app() first"
    with Session(ENGINE) as session:
        yield session


# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------


def compute_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def clean_text(text: str) -> str:
    """Clean text by normalizing Unicode and removing non-printable characters.
    
    Handles common issues from AI models like \u2011, \u00f3, etc.
    """
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    
    # Normalize Unicode to decomposed form, then recompose
    text = unicodedata.normalize('NFKC', text)
    
    # Replace problematic Unicode characters
    replacements = {
        '\u2011': '-',  # Non-breaking hyphen
        '\u2013': '-',  # En dash
        '\u2014': '--', # Em dash
        '\u2018': "'",  # Left single quotation mark
        '\u2019': "'",  # Right single quotation mark
        '\u201C': '"',  # Left double quotation mark
        '\u201D': '"',  # Right double quotation mark
        '\u2026': '...', # Horizontal ellipsis
        '\u00A0': ' ',  # Non-breaking space
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Remove other non-printable characters but keep common whitespace
    text = ''.join(char for char in text if unicodedata.category(char)[0] not in 'C' or char in '\t\n\r ')
    
    # Clean up multiple spaces and trim
    text = ' '.join(text.split())
    
    return text


def derive_author_title_from_path(epub_path: Path) -> Tuple[Optional[str], Optional[str]]:
    # Expect: library/Author_with_underscores/Title_with_underscores.epub
    try:
        author_dir = epub_path.parent.name
        title_file = epub_path.stem
        author = author_dir.replace("_", " ") if author_dir else None
        title = title_file.replace("_", " ") if title_file else None
        return author, title
    except Exception:
        return None, None


def _repair_epub_for_missing_resources(epub_path: Path) -> Optional[Path]:
    """Create a repaired temporary copy of the EPUB where missing image/font entries referenced
    from the OPF are replaced with tiny placeholder files so parsers like ebooklib don't fail.

    Returns path to the temporary EPUB file or None if repair isn't needed/failed.
    """
    try:
        with zipfile.ZipFile(epub_path, 'r') as zf:
            names = set(zf.namelist())
            # Try to read container and OPF
            try:
                container_xml = zf.read('META-INF/container.xml')
                root = ET.fromstring(container_xml)
                ns = {'c': 'urn:oasis:names:tc:opendocument:xmlns:container'}
                rootfile_el = root.find('.//c:rootfile', ns)
                opf_path = rootfile_el.attrib.get('full-path') if rootfile_el is not None else None
            except Exception:
                opf_path = None

            missing: List[str] = []
            if opf_path:
                try:
                    opf_data = zf.read(opf_path)
                    opf_root = ET.fromstring(opf_data)
                    opf_dir = posixpath.dirname(opf_path)
                    # Inspect manifest for hrefs
                    for it in opf_root.findall('.//{http://www.idpf.org/2007/opf}item'):
                        href = it.attrib.get('href')
                        if href:
                            full = posixpath.normpath(posixpath.join(opf_dir, href)) if opf_dir else href
                            if full not in names:
                                missing.append(full)
                except Exception:
                    pass

            # If nothing missing, return None (no repair needed)
            if not missing:
                return None

            # Create temp dir and copy all entries, adding placeholders for missing files
            tmpd = Path(tempfile.mkdtemp(prefix='skald_epub_repair_'))
            tmp_epub = tmpd / (epub_path.stem + '.epub')
            with zipfile.ZipFile(tmp_epub, 'w') as outz:
                for n in names:
                    try:
                        data = zf.read(n)
                        outz.writestr(n, data)
                    except Exception:
                        # skip problematic entries
                        continue
                # Add tiny placeholders for missing resources
                for m in missing:
                    # Determine extension
                    ext = Path(m).suffix.lower()
                    if ext in ('.jpg', '.jpeg', '.png'):
                        # create a 1x1 transparent PNG base64
                        png1x1 = base64.b64decode(
                            'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII='
                        )
                        outz.writestr(m, png1x1)
                    elif ext in ('.ttf', '.otf'):
                        # Create a minimal empty TTF placeholder (not a valid font but prevents missing-file errors)
                        outz.writestr(m, b'')
                    else:
                        outz.writestr(m, b'')
            return tmp_epub
    except Exception:
        return None


def parse_epub_metadata(epub_path: Path) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    # Primary path: use ebooklib
    try:
        book = epub.read_epub(epub_path.as_posix())
        title_list = book.get_metadata("DC", "title")
        title = title_list[0][0] if title_list else None
        creators = book.get_metadata("DC", "creator")
        author = creators[0][0] if creators else None
        languages = book.get_metadata("DC", "language")
        language = languages[0][0] if languages else None
        return title, author, language
    except Exception as e:
        LOGGER.warning("ebooklib failed to parse EPUB metadata for %s: %s; attempting repair and XML fallback", epub_path, e)
        # Try a light repair: if the EPUB references missing font/image files, insert small placeholders
        try:
            repaired = _repair_epub_for_missing_resources(epub_path)
            if repaired:
                try:
                    book = epub.read_epub(repaired.as_posix())
                    title_list = book.get_metadata("DC", "title")
                    title = title_list[0][0] if title_list else None
                    creators = book.get_metadata("DC", "creator")
                    author = creators[0][0] if creators else None
                    languages = book.get_metadata("DC", "language")
                    language = languages[0][0] if languages else None
                    return title, author, language
                except Exception:
                    # If repair didn't help, continue to ZIP/XML fallback below
                    pass
        except Exception:
            # Repair attempts should not raise; if they do, ignore and continue to fallback
            pass
    # Fallback: read container.xml and OPF directly from ZIP
    try:
        with zipfile.ZipFile(epub_path, 'r') as zf:
            container_xml = zf.read('META-INF/container.xml')
            root = ET.fromstring(container_xml)
            ns = {'c': 'urn:oasis:names:tc:opendocument:xmlns:container'}
            rootfile_el = root.find('.//c:rootfile', ns)
            if rootfile_el is None:
                return None, None, None
            opf_path = rootfile_el.attrib.get('full-path')
            if not opf_path:
                return None, None, None
            opf_data = zf.read(opf_path)
            opf_root = ET.fromstring(opf_data)
            # Try to find DC elements regardless of metadata namespace wrapping
            def _find_text(tag: str) -> Optional[str]:
                el = opf_root.find(f'.//{{http://purl.org/dc/elements/1.1/}}{tag}')
                return el.text.strip() if el is not None and el.text else None
            title = _find_text('title')
            author = _find_text('creator')
            language = _find_text('language')
            return title, author, language
    except Exception as e2:
        LOGGER.warning("ZIP/XML metadata fallback failed for %s: %s", epub_path, e2)
        return None, None, None


ALLOWED_TAGS = set(
    [
        "a",
        "abbr",
        "acronym",
        "b",
        "blockquote",
        "code",
        "em",
        "i",
        "li",
        "ol",
        "strong",
        "ul",
        "p",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "br",
        "hr",
        "span",
        "div",
    ]
)
ALLOWED_ATTRS = {"a": ["href", "title"], "*": ["class", "id"]}


def epub_to_light_html(epub_path: Path, max_chars: int = 200_000) -> str:
    def sanitize_join(html_parts: List[str]) -> str:
        combined = "\n".join(html_parts)
        return bleach.clean(combined, tags=ALLOWED_TAGS, attributes=ALLOWED_ATTRS, strip=True)

    # First try with ebooklib (fast path)
    try:
        book = epub.read_epub(epub_path.as_posix())
        parts: List[str] = []
        count = 0
        for item in book.get_items():
            try:
                if item.get_type() == ITEM_DOCUMENT:
                    soup = BeautifulSoup(item.get_content(), "lxml")
                    # Remove scripts/styles
                    for tag in soup(["script", "style"]):
                        tag.decompose()
                    body = soup.body or soup
                    text_html = str(body)
                    parts.append(text_html)
                    count += len(text_html)
                    if count >= max_chars:
                        break
            except Exception as ie:
                # Skip problematic items and continue
                LOGGER.warning("Skipping EPUB item due to error: %s", ie)
                continue
        if parts:
            return sanitize_join(parts)
    except Exception as e:
        LOGGER.warning("ebooklib failed to read EPUB %s: %s; attempting lightweight repair and falling back to zip scan", epub_path, e)
        # Attempt repair copy with placeholders for missing resources and retry once
        try:
            repaired = _repair_epub_for_missing_resources(epub_path)
            if repaired:
                try:
                    book = epub.read_epub(repaired.as_posix())
                    parts = []
                    count = 0
                    for item in book.get_items():
                        try:
                            if item.get_type() == ITEM_DOCUMENT:
                                soup = BeautifulSoup(item.get_content(), "lxml")
                                for tag in soup(["script", "style"]):
                                    tag.decompose()
                                body = soup.body or soup
                                text_html = str(body)
                                parts.append(text_html)
                                count += len(text_html)
                                if count >= max_chars:
                                    break
                        except Exception as ie:
                            LOGGER.warning("Skipping EPUB item due to error after repair: %s", ie)
                            continue
                    if parts:
                        return sanitize_join(parts)
                except Exception:
                    pass
        except Exception:
            pass

    # Fallback: read HTML files directly from the ZIP, ignoring missing resources
    try:
        parts: List[str] = []
        count = 0
        with zipfile.ZipFile(epub_path, 'r') as zf:
            # Collect HTML-like entries
            names = [n for n in zf.namelist() if n.lower().endswith((".xhtml", ".html", ".htm"))]
            for name in names:
                try:
                    data = zf.read(name)
                    soup = BeautifulSoup(data, "lxml")
                    for tag in soup(["script", "style"]):
                        tag.decompose()
                    body = soup.body or soup
                    text_html = str(body)
                    parts.append(text_html)
                    count += len(text_html)
                    if count >= max_chars:
                        break
                except KeyError as ke:
                    # Missing file entries: skip
                    LOGGER.warning("Missing file inside EPUB (%s): %s", name, ke)
                except Exception as e2:
                    LOGGER.warning("Skipping HTML entry due to error (%s): %s", name, e2)
                    continue
        if parts:
            return sanitize_join(parts)
        # As last resort, return minimal notice
        return "<div><p>Unable to render EPUB content.</p></div>"
    except Exception as e3:
        LOGGER.error("Failed to render EPUB via zip fallback %s: %s", epub_path, e3)
        raise


# ------------------------------------------------------------
# Covers (extraction and fetch) 🖼️
# ------------------------------------------------------------


def _cover_cache_path_for_sha(sha256: str) -> Optional[Path]:
    if not sha256:
        return None
    base = (COVERS_DIR or (Path.cwd() / "covers"))
    # Check existing known extensions
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".svg"):
        p = base / f"{sha256}{ext}"
        if p.exists():
            return p
    # Negative cache marker
    neg = base / f"{sha256}.none"
    if neg.exists():
        return None
    # Default jpg path to write to
    return base / f"{sha256}.jpg"


def _thumb_cache_path_for_sha(sha256: str) -> Optional[Path]:
    if not sha256:
        return None
    base = (COVERS_DIR or (Path.cwd() / "covers"))
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        p = base / f"{sha256}.thumb{ext}"
        if p.exists():
            return p
    if (base / f"{sha256}.none").exists():
        return None
    return base / f"{sha256}.thumb.jpg"


def _save_cover_bytes(sha256: str, data: bytes, suggested_ext: Optional[str] = None) -> Path:
    base = (COVERS_DIR or (Path.cwd() / "covers"))
    base.mkdir(parents=True, exist_ok=True)
    ext = (suggested_ext or ".jpg").lower()
    if not ext.startswith('.'):
        ext = "." + ext
    # Normalize common content types
    if ext in (".jpeg", ".jpg", ".png", ".webp", ".svg"):
        pass
    else:
        ext = ".jpg"
    path = base / f"{sha256}{ext}"
    path.write_bytes(data)
    return path


def _ensure_thumbnail(sha256: str, max_w: int = 200, max_h: int = 300) -> Optional[Path]:
    try:
        try:
            from PIL import Image  # type: ignore
        except Exception:
            return None
        full = _cover_cache_path_for_sha(sha256)
        if not full or not full.exists():
            return None
        thumb_path = _thumb_cache_path_for_sha(sha256) or (COVERS_DIR or (Path.cwd() / "covers")) / f"{sha256}.thumb.jpg"
        if thumb_path.exists():
            return thumb_path
        with Image.open(full.as_posix()) as im:
            im = im.convert("RGB")
            im.thumbnail((max_w, max_h))
            thumb_path.parent.mkdir(parents=True, exist_ok=True)
            im.save(thumb_path.as_posix(), format="JPEG", quality=82)
        return thumb_path
    except Exception as e:
        LOGGER.warning("Thumbnail generation failed for %s: %s", sha256, e)
        return None


def _cache_headers_for(path: Path) -> Dict[str, str]:
    try:
        data = path.read_bytes()
    except Exception:
        data = b""
    etag = hashlib.md5(data).hexdigest() if data else hashlib.md5(path.name.encode("utf-8", errors="ignore")).hexdigest()
    return {
        "Cache-Control": "public, max-age=31536000, immutable",
        "ETag": f'W/"{etag}"',
    }


def _mark_cover_missing(sha256: str) -> None:
    base = (COVERS_DIR or (Path.cwd() / "covers"))
    base.mkdir(parents=True, exist_ok=True)
    (base / f"{sha256}.none").write_text("", encoding="utf-8")


def _guess_mime_from_path(p: Path) -> str:
    mime, _ = mimetypes.guess_type(p.name)
    return mime or "image/jpeg"


def try_extract_cover_from_epub(epub_path: Path) -> Optional[Tuple[bytes, str]]:
    """Try to extract a cover image from the EPUB file.

    Returns (bytes, ext) or None.
    """
    try:
        with zipfile.ZipFile(epub_path, 'r') as zf:
            # 1) Read container.xml to find OPF
            try:
                container_xml = zf.read('META-INF/container.xml')
                root = ET.fromstring(container_xml)
                ns = {'c': 'urn:oasis:names:tc:opendocument:xmlns:container'}
                rootfile_el = root.find('.//c:rootfile', ns)
                opf_path = rootfile_el.attrib.get('full-path') if rootfile_el is not None else None
            except Exception:
                opf_path = None

            manifest: Dict[str, Dict[str, str]] = {}
            cover_href: Optional[str] = None
            opf_dir = ""
            if opf_path:
                opf_dir = posixpath.dirname(opf_path)
                try:
                    opf_data = zf.read(opf_path)
                    opf_root = ET.fromstring(opf_data)
                    # Build manifest map id -> href, properties
                    for it in opf_root.findall('.//{http://www.idpf.org/2007/opf}item'):
                        iid = it.attrib.get('id')
                        href = it.attrib.get('href')
                        props = it.attrib.get('properties', '')
                        if iid and href:
                            manifest[iid] = {"href": href, "properties": props}
                            if 'cover-image' in props:
                                cover_href = href
                    # meta name="cover" content="id"
                    if not cover_href:
                        for meta in opf_root.findall('.//{http://www.idpf.org/2007/opf}meta'):
                            if meta.attrib.get('name') == 'cover':
                                cover_id = meta.attrib.get('content')
                                if cover_id and cover_id in manifest:
                                    cover_href = manifest[cover_id]['href']
                                    break
                except Exception:
                    pass

            # If we found a cover href, resolve and read it
            if cover_href:
                cover_name = posixpath.normpath(posixpath.join(opf_dir, cover_href)) if opf_dir else cover_href
                try:
                    data = zf.read(cover_name)
                    ext = Path(cover_name).suffix.lower().lstrip('.') or 'jpg'
                    return data, ext
                except Exception:
                    pass

            # Fallback: pick best-looking image by heuristic (name contains 'cover' or largest)
            image_exts = ('.jpg', '.jpeg', '.png', '.webp')
            image_names = [n for n in zf.namelist() if n.lower().endswith(image_exts)]
            if not image_names:
                return None
            # Prefer names with 'cover'
            cover_candidates = [n for n in image_names if 'cover' in n.lower()]
            names_to_consider = cover_candidates or image_names
            # Choose largest by size
            sizes = {}
            for n in names_to_consider:
                try:
                    info = zf.getinfo(n)
                    sizes[n] = info.file_size
                except Exception:
                    sizes[n] = 0
            best = max(names_to_consider, key=lambda n: sizes.get(n, 0))
            try:
                data = zf.read(best)
                ext = Path(best).suffix.lower().lstrip('.') or 'jpg'
                return data, ext
            except Exception:
                return None
    except Exception:
        return None


def try_fetch_cover_via_openlibrary(author: str, title: str, timeout: float = 8.0) -> Optional[bytes]:
    try:
        global _COVER_LAST_REQUEST
        # Basic global throttle
        with _COVER_FETCH_LOCK:
            import time as _t
            now = _t.monotonic()
            wait = max(0.0, (_COVER_MIN_INTERVAL_MS/1000.0) - (now - _COVER_LAST_REQUEST))
            if wait > 0:
                _t.sleep(wait)
        params = {"q": f"title:{title} author:{author}", "fields": "cover_i,cover_edition_key,key", "limit": 1}
        url = "https://openlibrary.org/search.json"
        with httpx.Client(timeout=timeout) as client:
            r = client.get(url, params=params)
            r.raise_for_status()
            data = r.json()
            docs = data.get('docs') or []
            if not docs:
                return None
            doc = docs[0]
            cover_i = doc.get('cover_i')
            if not cover_i:
                # Fallback: use OL key (work/edition) if present
                olid = (doc.get('cover_edition_key') or '').strip()
                if olid:
                    img_url = f"https://covers.openlibrary.org/b/olid/{olid}-L.jpg"
                else:
                    return None
            else:
                img_url = f"https://covers.openlibrary.org/b/id/{cover_i}-L.jpg"
            # Throttle again before image
            with _COVER_FETCH_LOCK:
                now2 = _t.monotonic()
                wait2 = max(0.0, (_COVER_MIN_INTERVAL_MS/1000.0) - (now2 - _COVER_LAST_REQUEST))
                if wait2 > 0:
                    _t.sleep(wait2)
            ir = client.get(img_url)
            with _COVER_FETCH_LOCK:
                _COVER_LAST_REQUEST = _t.monotonic()
            if ir.status_code == 200 and ir.headers.get('content-type','').startswith('image'):
                return ir.content
    except Exception:
        return None
    return None


def try_fetch_cover_via_google(author: str, title: str, timeout: float = 8.0) -> Optional[bytes]:
    try:
        global _COVER_LAST_REQUEST
        # Basic global throttle
        with _COVER_FETCH_LOCK:
            import time as _t
            now = _t.monotonic()
            wait = max(0.0, (_COVER_MIN_INTERVAL_MS/1000.0) - (now - _COVER_LAST_REQUEST))
            if wait > 0:
                _t.sleep(wait)
        q = f"intitle:{title} inauthor:{author}"
        url = "https://www.googleapis.com/books/v1/volumes"
        with httpx.Client(timeout=timeout) as client:
            r = client.get(url, params={"q": q, "maxResults": 1})
            r.raise_for_status()
            data = r.json()
            items = data.get('items') or []
            if not items:
                return None
            vi = (items[0].get('volumeInfo') or {})
            links = vi.get('imageLinks') or {}
            # Prefer largest link available
            for k in ("extraLarge", "large", "medium", "small", "thumbnail", "smallThumbnail"):
                u = links.get(k)
                if u:
                    # Throttle again before image
                    with _COVER_FETCH_LOCK:
                        now2 = _t.monotonic()
                        wait2 = max(0.0, (_COVER_MIN_INTERVAL_MS/1000.0) - (now2 - _COVER_LAST_REQUEST))
                        if wait2 > 0:
                            _t.sleep(wait2)
                    ir = client.get(u)
                    with _COVER_FETCH_LOCK:
                        _COVER_LAST_REQUEST = _t.monotonic()
                    if ir.status_code == 200 and ir.headers.get('content-type','').startswith('image'):
                        return ir.content
    except Exception:
        return None
    return None


def get_or_build_cover(book: "Book") -> Optional[Path]:
    """Return a path to a cached cover image for this book, creating it if needed."""
    cache_path = _cover_cache_path_for_sha(book.sha256)
    if cache_path and cache_path.exists():
        return cache_path

    # Try extract from EPUB
    data_ext = try_extract_cover_from_epub(Path(book.path))
    if data_ext is not None:
        data, ext = data_ext
        return _save_cover_bytes(book.sha256, data, ext)

    # Try OpenLibrary by author/title
    data = try_fetch_cover_via_openlibrary(book.author, book.title, timeout=CONFIG.openlibrary_timeout)
    if data:
        return _save_cover_bytes(book.sha256, data, ".jpg")

    # Try Google Books
    data = try_fetch_cover_via_google(book.author, book.title, timeout=CONFIG.googlebooks_timeout)
    if data:
        return _save_cover_bytes(book.sha256, data, ".jpg")
    _mark_cover_missing(book.sha256)
    return None


# ------------------------------------------------------------
# Indexer
# ------------------------------------------------------------


def index_library(session: Session, library_path: Path) -> Dict[str, Any]:
    added = 0
    skipped = 0
    updated = 0
    errors: List[str] = []

    library_path = library_path.expanduser().resolve()
    library_path.mkdir(parents=True, exist_ok=True)

    for epub_path in library_path.rglob("*.epub"):
        try:
            sha = compute_sha256(epub_path)
            existing = session.exec(select(Book).where(Book.sha256 == sha)).first()
            stat = epub_path.stat()
            size_bytes = stat.st_size
            modified_iso = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
            title, author, language = parse_epub_metadata(epub_path)
            if not title or not author:
                d_author, d_title = derive_author_title_from_path(epub_path)
                author = author or d_author or "Unknown"
                title = title or d_title or epub_path.stem

            if existing:
                # Update minimal fields if path or timestamps changed
                changed = False
                if (
                    existing.path != epub_path.as_posix()
                    or existing.size_bytes != size_bytes
                    or existing.modified_iso != modified_iso
                    or existing.title != title
                    or existing.author != author
                    or existing.language != language
                ):
                    existing.path = epub_path.as_posix()
                    existing.size_bytes = size_bytes
                    existing.modified_iso = modified_iso
                    existing.title = title
                    existing.author = author
                    existing.language = language
                    existing.updated_at = datetime.now(timezone.utc)
                    session.add(existing)
                    updated += 1
                    changed = True
                if not changed:
                    skipped += 1
                continue

            book = Book(
                title=title or epub_path.stem,
                author=author or "Unknown",
                language=language,
                path=epub_path.as_posix(),
                size_bytes=size_bytes,
                modified_iso=modified_iso,
                sha256=sha,
            )
            session.add(book)
            added += 1
        except Exception as e:
            LOGGER.exception("Index error for %s: %s", epub_path, e)
            errors.append(f"{epub_path}: {e}")

    session.commit()
    return {"added": added, "updated": updated, "skipped": skipped, "errors": errors}


# ------------------------------------------------------------
# LLM Client (LMStudio)
# ------------------------------------------------------------


class LMClient:
    def __init__(
        self,
        base_url: str,
        timeout: float = 300.0,  # Aumentado a 60 segundos para modelos lentos
        enabled: bool = False,
        mock: bool = False,
        model: Optional[str] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.enabled = enabled
        self.mock = mock
        self.model = model or "openai/gpt-oss-20"
        # Try to load a JSON template to include in prompts (optional)
        self._template_json_str: Optional[str] = None
        try:
            tpl_path = Path(__file__).parent / "enrichment_model.json"
            if tpl_path.exists():
                self._template_json_str = tpl_path.read_text(encoding="utf-8").strip()
        except Exception as e:
            LOGGER.debug("Could not load enrichment_model.json: %s", e)

    def _build_simple_prompt(self, author: str, title: str, lang: Optional[str] = None) -> str:
        """Minimal prompt for basic enrichment with language control.

        Keeps the expected JSON tiny and easy for local models to follow.
        """
        target_lang = (lang or CONFIG.default_language or "es").strip()
        example_str = (
            "{\n"
            "  \"genre\": \"Non-fiction\",\n"
            "  \"year\": 2002,\n"
            "  \"audience\": \"adult\",\n"
            "  \"confidence\": 0.8,\n"
            "  \"tags\": [\"science\", \"astronomy\", \"physics\", \"cosmology\", \"history\"],\n"
            "  \"premise\": \"Brief summary without spoilers...\"\n"
            "}"
        )

        return (
            "Analyze this book and return basic metadata as JSON.\n\n"
            f"Author: \"{author}\"\n"
            f"Title: \"{title}\"\n\n"
            f"Write ALL fields (including tags and the summary) in this language: \"{target_lang}\".\n"
            "Do NOT include your reasoning. Only return the final JSON object.\n\n"
            "Return ONLY a valid JSON object matching this structure (replace values with real data or null only if truly unknown):\n"
            f"{example_str}\n\n"
            "Rules:\n"
            "- REQUIRED: provide genre, at least 4 tags, and a summary (premise) in the target language\n"
            "- audience must be one of: \"children\", \"young_adult\", \"adult\", \"general\"\n"
            "- confidence between 0.0 and 1.0 (certainty)\n"
            "- tags are short keywords (3-8 items)\n"
            "- summary: 120-200 words, spoiler-free\n"
            "- NO explanations, ONLY the JSON"
        )

    def _example_from_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Derive a minimal example JSON object from a JSON Schema.

        Focuses on the top-level object's properties; picks reasonable
        placeholder values and keeps shape consistent. Unknowns default to null/empty.
        """
        def pick_example(prop_schema: Dict[str, Any]) -> Any:
            t = prop_schema.get("type")
            # Normalize 'type' as list for easier handling
            types: List[str] = []
            if isinstance(t, list):
                types = [str(x) for x in t]
            elif isinstance(t, str):
                types = [t]

            # Prefer concrete type if union includes null
            non_null = [x for x in types if x != "null"]
            main_t = non_null[0] if non_null else (types[0] if types else None)

            if isinstance(prop_schema.get("enum"), list) and prop_schema.get("enum"):
                # Choose the first enum option for example
                return prop_schema["enum"][0]

            # If nullable, prefer showing null in the example to signal unknown allowed
            if "null" in types:
                return None

            if main_t == "string":
                return ""
            if main_t == "integer":
                return 0
            if main_t == "number":
                return 0.0
            if main_t == "array":
                items = prop_schema.get("items") or {}
                # Provide one example element only if items has enum; otherwise empty
                if isinstance(items, dict) and isinstance(items.get("enum"), list) and items.get("enum"):
                    return [items["enum"][0]]
                return []
            if main_t == "object":
                # Recurse one level for nested object
                if isinstance(prop_schema.get("properties"), dict):
                    return {k: pick_example(v) for k, v in prop_schema["properties"].items()}
                return {}
            # Default for null/unknown types
            return None

        if not isinstance(schema, dict) or schema.get("type") != "object":
            return {}
        props = schema.get("properties")
        if not isinstance(props, dict):
            return {}
        example: Dict[str, Any] = {}
        for key, prop_schema in props.items():
            if isinstance(prop_schema, dict):
                example[key] = pick_example(prop_schema)
        return example

    def _mock_enrichment(self) -> Dict[str, Any]:
        """Mock enrichment data for testing."""
        return {
            "genre": "Unknown",
            "year": None,
            "audience": "general",
            "confidence": 0.5,
            "tags": ["mock"],
            "premise": "Mock enrichment data for testing.",
        }

    def enrich_simple(self, author: str, title: str, lang: Optional[str] = None) -> Tuple[Optional[Dict[str, Any]], str]:
        """Simplified enrichment method."""
        if not self.enabled:
            return None, "disabled"
        if not self.model:
            return None, "no_model"
        
        if self.mock:
            return self._mock_enrichment(), "ok"
        
        now_iso = datetime.now(timezone.utc).isoformat()
        tries = 3
        
        # Simple URL construction
        base_url = self.base_url.rstrip('/')
        if not base_url.endswith('/v1/chat/completions'):
            base_url = base_url + '/v1/chat/completions'
            
        LOGGER.info("LM enrich for '%s' by %s using %s", title, author, base_url)
        
        for attempt in range(1, tries + 1):
            try:
                prompt = self._build_simple_prompt(author, title, lang=lang)
                
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are a librarian. Respond with ONLY valid JSON, no explanations."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 600
                }
                
                LOGGER.info("LM prompt sent:\n%s", prompt)
                
                with httpx.Client(timeout=self.timeout) as client:
                    with lm_rate_limit():
                        resp = client.post(base_url, json=payload, 
                                         headers={"Content-Type": "application/json"})
                
                LOGGER.info("LM response status: %d", resp.status_code)
                
                if resp.status_code == 200:
                    data = resp.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    LOGGER.info("LM raw response from model:\n--- RAW RESPONSE START ---\n%s\n--- RAW RESPONSE END ---", content)
                    
                    # Parse JSON
                    try:
                        # Prefer the last JSON object in the content (models with reasoning may emit multiple)
                        text = content if isinstance(content, str) else str(content)
                        import re
                        json_candidates = re.findall(r"\{[\s\S]*?\}", text)
                        if json_candidates:
                            candidate = json_candidates[-1]
                            result = json.loads(candidate)
                        else:
                            result = json.loads(text)
                        LOGGER.info("LM parsed JSON data:\n--- PARSED JSON START ---\n%s\n--- PARSED JSON END ---", json.dumps(result, indent=2, ensure_ascii=False))
                    except json.JSONDecodeError:
                        # Try to extract JSON from text
                        import re
                        json_match = re.search(r'\{[\s\S]*\}', content, re.DOTALL)
                        if json_match:
                            extracted_json = json_match.group()
                            LOGGER.info("LM extracted JSON from text:\n--- EXTRACTED JSON START ---\n%s\n--- EXTRACTED JSON END ---", extracted_json)
                            result = json.loads(extracted_json)
                            LOGGER.info("LM parsed extracted JSON:\n--- FINAL JSON START ---\n%s\n--- FINAL JSON END ---", json.dumps(result, indent=2, ensure_ascii=False))
                        else:
                            LOGGER.error("LM could not extract JSON from response: %s", content[:500])
                            raise ValueError("No valid JSON found in response")
                    
                    # Basic validation
                    if not isinstance(result, dict):
                        raise ValueError("Response is not a JSON object")

                    # Clean Unicode and non-printable characters from text fields
                    for key, value in result.items():
                        if isinstance(value, str):
                            result[key] = clean_text(value)
                        elif isinstance(value, list):
                            result[key] = [clean_text(item) if isinstance(item, str) else item for item in value]

                    # Light checks for richer content
                    def is_thin(r: Dict[str, Any]) -> bool:
                        genre_ok = bool((r.get("genre") or "").strip())
                        premise_len = len((r.get("premise") or "").strip())
                        tags_ok = isinstance(r.get("tags"), list) and len(r.get("tags") or []) >= 3
                        return not (genre_ok and tags_ok and premise_len >= 80)

                    if is_thin(result) and attempt < tries:
                        LOGGER.info("LM result seems too thin; attempting one corrective retry")
                        # Nudge by appending a short corrective hint to the user message
                        payload["messages"][1]["content"] += "\n\nPlease include a concrete 'genre', at least 4 relevant 'tags', and a 120-200 word 'premise' in the target language."
                        with httpx.Client(timeout=self.timeout) as client:
                            with lm_rate_limit():
                                resp2 = client.post(base_url, json=payload, headers={"Content-Type": "application/json"})
                        if resp2.status_code == 200:
                            data2 = resp2.json()
                            content2 = data2.get("choices", [{}])[0].get("message", {}).get("content", "")
                            try:
                                result2 = json.loads(content2) if isinstance(content2, str) else content2
                                if isinstance(result2, dict) and not is_thin(result2):
                                    result = result2
                            except Exception:
                                pass
                    
                    # Add metadata
                    result["enriched_by"] = f"LMStudio:{self.model}"
                    result["enriched_at"] = now_iso
                    
                    LOGGER.info("LM enrich successful - Final result:\n--- FINAL RESULT START ---\n%s\n--- FINAL RESULT END ---", json.dumps(result, indent=2, ensure_ascii=False))
                    return result, "ok"
                else:
                    LOGGER.warning("LM error %d: %s", resp.status_code, resp.text[:200])
                    
            except Exception as e:
                LOGGER.warning("LM attempt %d failed: %s", attempt, str(e))
                if attempt < tries:
                    time.sleep(2 * attempt)  # Simple backoff
        
        LOGGER.error("LM enrich failed after %d attempts", tries)
        return None, "failed"

    def enrich(self, author: str, title: str) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Returns (payload, status). status in {"ok", "failed"}
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        if not self.enabled:
            return None, "failed"
        if self.mock:
            payload: Dict[str, Any] = {
                "genre": "Unknown",
                "year": None,
                "series": None,
                "series_number": None,
                "audience": "general",
                "tags": ["mock"],
                "content_warnings": [],
                "premise": f"Autogenerated summary for '{title}' by {author}.",
                "confidence": 0.5,
                "enriched_by": "LMStudio-mock",
                "enriched_at": now_iso,
            }
            return payload, "ok"

        tries = 3  # Reducido a 3 intentos pero con timeout mayor
        last_err: Optional[Exception] = None
        # Determine OpenAI-style endpoint; if user gave root or other path, normalize to root + /v1/chat/completions
        is_openai_style = "/v1/chat/completions" in self.base_url
        if is_openai_style:
            openai_url = self.base_url
            # derive root for legacy
            p = urlparse(self.base_url)
            root = f"{p.scheme}://{p.netloc}" if (p.scheme and p.netloc) else self.base_url.split("/", 3)[0]
        else:
            p = urlparse(self.base_url)
            if p.scheme and p.netloc:
                root = f"{p.scheme}://{p.netloc}"
            else:
                # Fallback: assume given value is a host:port or root
                root = self.base_url.split("/", 1)[0]
            openai_url = root.rstrip("/") + "/v1/chat/completions"
        # We no longer use legacy /api/predict; always use OpenAI-style with schema
        LOGGER.debug("LM endpoint (openai-style)=%s", openai_url)
        for attempt in range(1, tries + 1):
            try:
                # Try OpenAI-style first
                LOGGER.info("LM enrich attempt %d/%d (openai-style) for '%s' by %s", attempt, tries, title, author)
                prompt = self._build_prompt(author, title)
                # schema = self._json_schema()
                LOGGER.info("LM JSON Schema sent to LMStudio: %s", json.dumps(prompt, indent=2))
                openai_payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": (
                            "You are an expert librarian and literary analyst. Your task is to enrich book metadata. "
                            "You MUST respond with ONLY valid JSON following the exact schema provided. "
                            "No additional text, no markdown, no code blocks, no explanations. "
                            "Required fields: audience (must be one of: children, young_adult, adult, general), "
                            "confidence (0.0-1.0 number). "
                            "If you don't know a field, use null for strings or [] for arrays. "
                            "Focus on accuracy over completeness."
                        )},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 800,  # Reducido para respuestas más concisas
                }
                LOGGER.debug("LM request payload: model=%s, timeout=%s, prompt_length=%d", 
                           self.model, self.timeout, len(prompt))
                LOGGER.info("LM prompt sent to LMStudio:\n--- SYSTEM MESSAGE ---\n%s\n--- USER MESSAGE ---\n%s\n--- END PROMPT ---", 
                           openai_payload["messages"][0]["content"], 
                           openai_payload["messages"][1]["content"])
                with httpx.Client(timeout=self.timeout) as client:
                    with lm_rate_limit():
                        resp = client.post(
                            openai_url,
                            json=openai_payload,
                            headers={"Content-Type": "application/json", "Accept": "application/json"},
                        )
                LOGGER.debug("LM response: status=%d, content_length=%d", resp.status_code, 
                           len(resp.content) if resp.content else 0)
                if resp.status_code == 200:
                    data = resp.json()
                    content = (data.get("choices", [{}])[0].get("message", {}).get("content", ""))
                    LOGGER.info("LM full response from LMStudio:\n--- RAW RESPONSE ---\n%s\n--- END RESPONSE ---", content)
                    
                    parsed = json.loads(content) if isinstance(content, str) else content
                    LOGGER.debug("LM parsed data: %s", str(parsed)[:200])
                    
                    payload = self._validate_payload(parsed)
                    LOGGER.debug("LM validated payload: %s", str(payload)[:200])
                    
                    # Quick sanity check: ensure we have required fields and some meaningful content
                    # audience and confidence are required by schema, others optional but at least one should be present
                    has_required = payload.get("audience") and payload.get("confidence") is not None
                    meaningful_keys = [
                        k for k in ("genre", "year", "series", "tags", "premise")
                        if payload.get(k) not in (None, "", [])
                    ]
                    LOGGER.debug("LM required fields present: %s, meaningful keys: %s", has_required, meaningful_keys)
                    
                    if not has_required:
                        LOGGER.warning("LM payload rejected - missing required fields (audience, confidence). Full payload: %s", payload)
                        raise ValueError("Missing required enrichment fields")
                    
                    if not meaningful_keys and not payload.get("premise"):
                        LOGGER.warning("LM payload rejected - no meaningful content. Full payload: %s", payload)
                        raise ValueError("Empty/meaningless enrichment payload")
                    
                    payload.setdefault("enriched_by", f"LMStudio:{self.model}")
                    payload.setdefault("enriched_at", now_iso)
                    LOGGER.info("LM enrich successful for '%s' by %s", title, author)
                    return payload, "ok"
                else:
                    # If server rejects schema, try text mode (no response_format) and parse loosely
                    if resp.status_code == 400 and "response_format" in (resp.text or ""):
                        openai_payload_text = dict(openai_payload)
                        openai_payload_text.pop("response_format", None)
                        with httpx.Client(timeout=self.timeout) as client:
                            with lm_rate_limit():
                                resp_text = client.post(
                                    openai_url,
                                    json=openai_payload_text,
                                    headers={"Content-Type": "application/json", "Accept": "application/json"},
                                )
                        if resp_text.status_code == 200:
                            data_t = resp_text.json()
                            content_t = (data_t.get("choices", [{}])[0].get("message", {}).get("content", ""))
                            parsed_t = self._parse_json_loose(content_t)
                            payload_t = self._validate_payload(parsed_t)
                            meaningful_keys_t = [
                                k for k in ("genre", "year", "series", "audience", "tags", "premise") if payload_t.get(k)
                            ]
                            if not meaningful_keys_t:
                                raise ValueError("Empty/meaningless enrichment payload (text mode)")
                            payload_t.setdefault("enriched_by", f"LMStudio:{self.model}")
                            payload_t.setdefault("enriched_at", now_iso)
                            return payload_t, "ok"
                    raise RuntimeError(f"OpenAI-style HTTP {resp.status_code}: {resp.text[:200]}")
            except (httpx.TimeoutException, httpx.ConnectTimeout, httpx.ReadTimeout) as e1:
                last_err = e1
                LOGGER.warning("LM enrich attempt %d (openai) timed out after %ds: %s", attempt, self.timeout, e1)
                # Don't try legacy on timeout, just retry with backoff
                if attempt < tries:
                    wait_time = min(5 * attempt, 15)  # Backoff: 5, 10, 15 segundos
                    LOGGER.info("Waiting %ds before retry...", wait_time)
                    time.sleep(wait_time)
                continue
            except Exception as e1:
                last_err = e1
                LOGGER.warning("LM enrich attempt %d (openai) failed: %s", attempt, e1)
                # No legacy fallback; keep last_err and continue retries
                last_err = e1
                LOGGER.debug("No legacy fallback; will retry openai-style if attempts remain")
            # Exponential backoff with jitter between attempts
            if attempt < tries:
                backoff = min(0.5 * (2 ** (attempt - 1)), 8.0)  # cap at 8s
                time.sleep(backoff + random.random() * 0.2)
        LOGGER.error("LM enrich failed after retries: %s", last_err)
        return None, "failed"

    def _build_prompt(self, author: str, title: str) -> str:
        # Schema-aligned prompt: embed the JSON Schema directly to avoid duplicates
        try:
            schema = self._json_schema()
        except Exception:
            schema = {}

        schema_str = json.dumps(schema, indent=2, ensure_ascii=False)

        instructions = (
            "You MUST return ONLY a JSON object (no markdown, no code fences, no explanations) that VALIDATES against the provided JSON Schema.\n"
            "If a field is unknown, use null (or [] for arrays). Keep tags concise."
        )

        return (
            f"Analyze this book and return metadata as JSON.\n\n"
            f"Author: \"{author}\"\n"
            f"Title: \"{title}\"\n\n"
            f"JSON Schema to follow:\n{schema_str}\n\n"
            f"Instructions:\n{instructions}"
        )

    @staticmethod
    def _parse_json_loose(text: str) -> Dict[str, Any]:
        """Extract JSON object from a text blob by locating the first '{' and last '}'."""
        if isinstance(text, dict):
            return text  # already a dict
        s = str(text)
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(s[start : end + 1])
            except Exception:
                pass
        return {}

    @staticmethod
    def _json_schema() -> Dict[str, Any]:
        """Load JSON Schema from enrichment_model.json file."""
        try:
            schema_path = Path(__file__).parent / "enrichment_model.json"
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema = json.load(f)
            return schema
        except Exception as e:
            LOGGER.warning("Failed to load enrichment_model.json, using fallback schema: %s", e)
            # Fallback minimal schema
            return {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "genre": {"type": ["string", "null"]},
                    "year": {"type": ["integer", "null"]},
                    "audience": {"type": "string", "enum": ["children", "young_adult", "adult", "general"]},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "premise": {"type": ["string", "null"]},
                },
                "required": ["audience", "confidence"],
                "additionalProperties": False,
            }

    @staticmethod
    def _validate_payload(data: Dict[str, Any]) -> Dict[str, Any]:
        # Accept expected keys; coerce types; pass-through localized
        expected = {
            "genre": (str, type(None)),
            "year": (int, str, type(None)),
            "series": (str, type(None)),
            "series_number": (int, str, type(None)),
            "audience": (str, type(None)),
            "tags": (list, type(None)),
            "content_warnings": (list, type(None)),
            "premise": (str, type(None)),
            "confidence": (int, float, str, type(None)),
            "enriched_by": (str, type(None)),
            "enriched_at": (str, type(None)),
            "localized": (dict, type(None)),
        }
        # Drop unknown keys to keep JSON tight
        out: Dict[str, Any] = {}
        for k, types in expected.items():
            v = data.get(k)
            if v is None:
                out[k] = None
                continue
            if not isinstance(v, types):
                out[k] = None
            else:
                out[k] = v
        # Numeric coercions
        if out.get("year") is not None:
            try:
                out["year"] = int(out["year"])  # type: ignore[index]
            except Exception:
                out["year"] = None
        if out.get("series_number") is not None:
            try:
                out["series_number"] = int(out["series_number"])  # type: ignore[index]
            except Exception:
                out["series_number"] = None
        if out.get("confidence") is not None:
            try:
                out["confidence"] = float(out["confidence"])  # type: ignore[index]
            except Exception:
                out["confidence"] = None
        # Lists to strings
        for list_key in ("tags", "content_warnings"):
            if out.get(list_key) is None:
                out[list_key] = []
            else:
                out[list_key] = [str(x) for x in (out[list_key] or [])]
        # Localized block
        loc = out.get("localized")
        if isinstance(loc, dict):
            cleaned: Dict[str, Any] = {}
            for lang in ("es", "en", "fr", "de", "pt"):
                blk = loc.get(lang)
                if isinstance(blk, dict):
                    b2: Dict[str, Any] = {}
                    if "premise" in blk:
                        b2["premise"] = None if blk["premise"] is None else str(blk["premise"])
                    if "tags" in blk:
                        b2["tags"] = [str(x) for x in (blk["tags"] or [])]
                    if b2:
                        cleaned[lang] = b2
            out["localized"] = cleaned if cleaned else None
        else:
            out["localized"] = None
        return out


LM = LMClient(CONFIG.lm_url, timeout=CONFIG.lm_timeout, enabled=CONFIG.lm_enabled, mock=CONFIG.lm_mock)
ENRICH_CHAIN: Optional[ChainEnricher] = None


def apply_enrichment_to_book(session: Session, book: Book, payload: Optional[Dict[str, Any]], status: str) -> Book:
    """Persist enrichment result.

    - On success: update enriched JSON and mirrored fields; set status=ok.
    - On failure: keep previous enriched data; set status=failed.
    """
    if status == "ok" and payload:
        book.enriched = payload
        book.enrichment_status = status
        book.genre = payload.get("genre")
        # year could be None or not an int
        year_val: Optional[int] = None
        y = payload.get("year")
        if isinstance(y, (int, str)):
            try:
                year_val = int(y)
            except Exception:
                year_val = None
        book.year = year_val
    else:
        # Only update the status; preserve previous data
        book.enrichment_status = status
    book.updated_at = datetime.now(timezone.utc)
    session.add(book)
    session.commit()
    session.refresh(book)
    return book


# ------------------------------------------------------------
# FastAPI App & Routes
# ------------------------------------------------------------


app = FastAPI(title="The Archive", version="0.1.0")

if CONFIG.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CONFIG.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}


@app.get("/")
def root() -> Response:
    """Serve index.html if present; otherwise, plain text."""
    index_path = Path(__file__).parent / "index.html"
    if index_path.exists():
        return FileResponse(index_path.as_posix(), media_type="text/html; charset=utf-8")
    return PlainTextResponse("The Archive API running. Open /docs for API docs.")


@app.get("/filters")
def get_filters(session: Session = Depends(get_session)) -> Dict[str, Any]:
    # Authors
    authors_rows = session.exec(select(Book.author).distinct().order_by(cast(Any, Book.author))).all()
    authors: List[str] = []
    for row in authors_rows:
        if isinstance(row, tuple) and row and isinstance(row[0], str):
            authors.append(row[0])
        elif isinstance(row, str):
            authors.append(row)
    # Genres (non-null)
    genres_rows = session.exec(select(Book.genre).where(cast(Any, Book.genre) != None).distinct().order_by(cast(Any, Book.genre))).all()  # noqa: E711
    genres: List[str] = []
    for row in genres_rows:
        if isinstance(row, tuple) and row and isinstance(row[0], str):
            genres.append(row[0])
        elif isinstance(row, str):
            genres.append(row)
    # Years
    year_rows = session.exec(select(Book.year).where(cast(Any, Book.year) != None)).all()  # noqa: E711
    years: List[int] = []
    for row in year_rows:
        # row might be a tuple like (year,) or a bare int
        if isinstance(row, tuple) and row and isinstance(row[0], int):
            years.append(row[0])
        elif isinstance(row, int):
            years.append(row)
    min_year: Optional[int] = min(years) if years else None
    current_year = datetime.now(timezone.utc).year
    return {
        "authors": authors,
        "genres": genres,
        "min_year": min_year,
        "current_year": current_year,
    }

@app.post("/config/reload")
def reload_config() -> Dict[str, Any]:
    """Reload config.json and rebuild LM client without full process restart."""
    global CONFIG, LM, DB_PATH, ENGINE
    old = CONFIG.model_dump()
    init_app()  # will reload config from file and rebuild LM
    new = CONFIG.model_dump()
    changed = {k: (old.get(k), new.get(k)) for k in new.keys() if old.get(k) != new.get(k)}
    return {"status": "ok", "changed": changed}


class PaginatedBooks(BaseModel):
    total: int
    page: int
    page_size: int
    items: List[Dict[str, Any]]


@app.get("/books", response_model=PaginatedBooks)
def list_books(
    q: Optional[str] = Query(default=None, description="Full-text query over title and author"),
    autor: Optional[str] = Query(default=None),
    titulo: Optional[str] = Query(default=None),
    genre: Optional[str] = Query(default=None),
    year_from: Optional[int] = Query(default=None, ge=0),
    year_to: Optional[int] = Query(default=None, ge=0),
    state_filter: Optional[str] = Query(default=None, description="Filter by state: favorite|read|pending"),
    sort_by: Optional[str] = Query(default="title", description="Sort by: title|author|year|added|progress|genre"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=CONFIG.page_size_default, ge=1, le=200),
    session: Session = Depends(get_session),
) -> PaginatedBooks:
    stmt = select(Book)
    if q:
        like = f"%{q}%"
        stmt = stmt.where((cast(Any, Book.title).ilike(like)) | (cast(Any, Book.author).ilike(like)))
    if autor:
        stmt = stmt.where(cast(Any, Book.author).ilike(f"%{autor}%"))
    if titulo:
        stmt = stmt.where(cast(Any, Book.title).ilike(f"%{titulo}%"))
    if genre:
        stmt = stmt.where(cast(Any, Book.genre).ilike(f"%{genre}%"))
    if year_from is not None:
        stmt = stmt.where((cast(Any, Book.year) != None) & (cast(Any, Book.year) >= year_from))  # noqa: E711
    if year_to is not None:
        stmt = stmt.where((cast(Any, Book.year) != None) & (cast(Any, Book.year) <= year_to))  # noqa: E711

    # State filtering if provided
    if state_filter in {"favorite", "read", "pending"}:
        flag = {
            "favorite": BookState.favorite_flag,
            "read": BookState.read_flag,
            "pending": BookState.pending_flag,
        }[state_filter]
        sub = select(BookState.book_id).where(flag == True)  # noqa: E712
        stmt = stmt.where(cast(Any, Book.id).in_(sub))

    # Apply sorting
    if sort_by == "author":
        stmt = stmt.order_by(cast(Any, Book.author))
    elif sort_by == "year":
        stmt = stmt.order_by(cast(Any, Book.year).desc())
    elif sort_by == "added":
        stmt = stmt.order_by(cast(Any, Book.created_at).desc())
    elif sort_by == "genre":
        stmt = stmt.order_by(cast(Any, Book.genre))
    else:  # Default to title
        stmt = stmt.order_by(cast(Any, Book.title))

    total = len(session.exec(stmt).all())
    # pagination
    offset = (page - 1) * page_size
    stmt = stmt.offset(offset).limit(page_size)
    rows = session.exec(stmt).all()
    ids = [b.id for b in rows if b is not None]
    
    # Map states for these ids
    state_map: Dict[int, BookState] = {}
    if ids:
        st_rows = session.exec(select(BookState).where(cast(Any, BookState.book_id).in_(ids))).all()
        for st in st_rows:
            if st and st.book_id is not None:
                state_map[st.book_id] = st
    items = []
    for b in rows:
        if not b:
            continue
        st = state_map.get(b.id or -1)
        
        # Calculate progress percentage
        progress_percent = 0
        if st:
            if st.percent is not None:
                progress_percent = st.percent
            elif st.last_mode == 'paged' and st.page is not None:
                # For paged mode, we need to estimate based on current page
                # This is a rough estimate since we don't store total pages
                progress_percent = min(100, st.page * 5)  # Rough estimate
            elif st.scroll_top is not None:
                # For scroll mode, we need the scroll ratio
                # This is also an estimate since we don't store scroll height
                progress_percent = min(100, st.scroll_top / 1000 * 100) if st.scroll_top > 0 else 0
        
        # Get enriched data for tags and word count
        tags = []
        word_count = None
        if b.enriched:
            try:
                enriched_data = json.loads(b.enriched) if isinstance(b.enriched, str) else b.enriched
                tags = enriched_data.get('tags', [])
                word_count = enriched_data.get('word_count')
            except (json.JSONDecodeError, TypeError):
                pass
        
        # Compute cover availability lazily without generating if heavy
        # Use thumbnail endpoint by default for lists (smaller, cached aggressively)
        cover_url = f"/books/{b.id}/cover/thumb"
        items.append(
                {
                    "id": b.id,
                    "title": b.title,
                    "author": b.author,
                    "language": b.language,
                    "size_bytes": b.size_bytes,
                    "modified_iso": b.modified_iso,
                    "sha256": b.sha256,
                    "genre": b.genre,
                    "year": b.year,
                    "enrichment_status": b.enrichment_status,
                    "favorite": bool(st.favorite_flag) if st else False,
                    "read": bool(st.read_flag) if st else False,
                    "pending": bool(st.pending_flag) if st else False,
                    "progress_percent": round(progress_percent, 1),
                    "tags": tags,
                    "word_count": word_count,
                    "created_at": b.created_at.isoformat() if b.created_at else None,
                    "cover_url": cover_url,
                }
            )
    return PaginatedBooks(total=total, page=page, page_size=page_size, items=items)


@app.get("/books/{book_id}")
def get_book(book_id: int, session: Session = Depends(get_session)) -> Dict[str, Any]:
    book = session.get(Book, book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    # attach state
    st = session.exec(select(BookState).where(BookState.book_id == book.id)).first()
    # Add cover URLs
    cover_url = f"/books/{book.id}/cover/thumb"
    full_cover_url = f"/books/{book.id}/cover"
    return {
        "id": book.id,
        "title": book.title,
        "author": book.author,
        "language": book.language,
        "path": book.path,
        "size_bytes": book.size_bytes,
        "modified_iso": book.modified_iso,
        "sha256": book.sha256,
        "enriched": book.enriched,
        "enrichment_status": book.enrichment_status,
        "genre": book.genre,
        "year": book.year,
        "created_at": book.created_at,
        "updated_at": book.updated_at,
        "favorite": bool(st.favorite_flag) if st else False,
        "read": bool(st.read_flag) if st else False,
        "pending": bool(st.pending_flag) if st else False,
    "cover_url": cover_url,
    "full_cover_url": full_cover_url,
    }


@app.get("/books/{book_id}/cover")
def get_cover(book_id: int, session: Session = Depends(get_session)):
    book = session.get(Book, book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    path = get_or_build_cover(book)
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail="Cover not available")
    headers = _cache_headers_for(path)
    return FileResponse(
        path.as_posix(),
        media_type=_guess_mime_from_path(path),
        filename=f"{book.title}_cover{path.suffix}",
        headers=headers,
    )


@app.get("/books/{book_id}/cover/thumb")
def get_cover_thumb(book_id: int, session: Session = Depends(get_session)):
    book = session.get(Book, book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    # Ensure we have a cover first
    path = get_or_build_cover(book)
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail="Cover not available")
    thumb = _ensure_thumbnail(book.sha256)
    if not thumb or not thumb.exists():
        # Fallback to full image
        headers = _cache_headers_for(path)
        return FileResponse(
            path.as_posix(),
            media_type=_guess_mime_from_path(path),
            filename=f"{book.title}_cover{path.suffix}",
            headers=headers,
        )
    headers = _cache_headers_for(thumb)
    return FileResponse(
        thumb.as_posix(),
        media_type=_guess_mime_from_path(thumb),
        filename=f"{book.title}_thumb{thumb.suffix}",
        headers=headers,
    )


@app.get("/download/{book_id}")
def download_book(book_id: int, session: Session = Depends(get_session)):
    book = session.get(Book, book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    file_path = Path(book.path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found on disk")
    return FileResponse(
        file_path.as_posix(),
        media_type="application/epub+zip",
        filename=f"{book.title}.epub",
    )


@app.get("/open/{book_id}")
def open_book(book_id: int, session: Session = Depends(get_session)):
    book = session.get(Book, book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    file_path = Path(book.path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found on disk")
    try:
        html = epub_to_light_html(file_path)
        return Response(content=html, media_type="text/html; charset=utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------- Progress & State APIs --------------------


class ProgressRequest(BaseModel):
    mode: Optional[str] = Field(default=None)
    scroll_top: Optional[int] = Field(default=None, ge=0)
    page: Optional[int] = Field(default=None, ge=1)
    percent: Optional[float] = Field(default=None, ge=0, le=1)


@app.get("/books/{book_id}/progress")
def get_progress(book_id: int, session: Session = Depends(get_session)) -> Dict[str, Any]:
    b = session.get(Book, book_id)
    if not b:
        raise HTTPException(status_code=404, detail="Book not found")
    st = session.exec(select(BookState).where(BookState.book_id == book_id)).first()
    if not st:
        return {"book_id": book_id, "mode": None, "scroll_top": None, "page": None, "percent": None, "updated_at": None}
    return {
        "book_id": book_id,
        "mode": st.last_mode,
        "scroll_top": st.scroll_top,
        "page": st.page,
        "percent": st.percent,
        "updated_at": st.updated_at,
    }


@app.post("/books/{book_id}/progress")
def set_progress(book_id: int, req: ProgressRequest, session: Session = Depends(get_session)) -> Dict[str, Any]:
    b = session.get(Book, book_id)
    if not b:
        raise HTTPException(status_code=404, detail="Book not found")
    st = session.exec(select(BookState).where(BookState.book_id == book_id)).first()
    if not st:
        st = BookState(book_id=book_id)
    if req.mode:
        st.last_mode = req.mode
    if req.scroll_top is not None:
        st.scroll_top = int(req.scroll_top)
    if req.page is not None:
        st.page = int(req.page)
    if req.percent is not None:
        try:
            st.percent = float(req.percent)
        except Exception:
            st.percent = None
    st.updated_at = datetime.now(timezone.utc)
    session.add(st)
    session.commit()
    return {"status": "ok"}


class StateRequest(BaseModel):
    favorite: Optional[bool] = None
    read: Optional[bool] = None
    pending: Optional[bool] = None


@app.get("/books/{book_id}/state")
def get_state(book_id: int, session: Session = Depends(get_session)) -> Dict[str, Any]:
    b = session.get(Book, book_id)
    if not b:
        raise HTTPException(status_code=404, detail="Book not found")
    st = session.exec(select(BookState).where(BookState.book_id == book_id)).first()
    return {
        "book_id": book_id,
        "favorite": bool(st.favorite_flag) if st else False,
        "read": bool(st.read_flag) if st else False,
        "pending": bool(st.pending_flag) if st else False,
    }


@app.post("/books/{book_id}/state")
def set_state(book_id: int, req: StateRequest, session: Session = Depends(get_session)) -> Dict[str, Any]:
    b = session.get(Book, book_id)
    if not b:
        raise HTTPException(status_code=404, detail="Book not found")
    st = session.exec(select(BookState).where(BookState.book_id == book_id)).first()
    if not st:
        st = BookState(book_id=book_id)
    if req.favorite is not None:
        st.favorite_flag = bool(req.favorite)
    if req.read is not None:
        st.read_flag = bool(req.read)
    if req.pending is not None:
        st.pending_flag = bool(req.pending)
    st.updated_at = datetime.now(timezone.utc)
    session.add(st)
    session.commit()
    return {"status": "ok", "favorite": st.favorite_flag, "read": st.read_flag, "pending": st.pending_flag}


class ReindexRequest(BaseModel):
    mode: Optional[str] = Field(default="sync", description="sync|async")


@app.post("/reindex")
def reindex(req: ReindexRequest, background: BackgroundTasks, session: Session = Depends(get_session)):
    lib_path = Path(CONFIG.library_path)
    if req.mode == "async":
        def task():
            with Session(ENGINE) as s:
                res = index_library(s, lib_path)
                LOGGER.info("Async reindex result: %s", res)

        background.add_task(task)
        return {"status": "started"}
    res = index_library(session, lib_path)
    return res


class EnrichBatchRequest(BaseModel):
    ids: Optional[List[int]] = None
    language: Optional[str] = Field(default=None, description="Preferred language for summaries and labels")


@app.post("/enrich/batch")
def enrich_batch(req: EnrichBatchRequest, session: Session = Depends(get_session)):
    ids = req.ids
    updated: List[int] = []
    failed: List[int] = []
    limit = CONFIG.enrichment_batch_size
    books: List[Book]
    if ids:
        books_seq = session.exec(select(Book).where(cast(Any, Book.id).in_(ids))).all()
        books = [b for b in books_seq if b is not None]
    else:
        books_seq = session.exec(
            select(Book).where(Book.enrichment_status != "ok").limit(limit)
        ).all()
        books = list(books_seq)
    for b in books:
        lang = req.language or CONFIG.default_language
        result: EnrichmentResult
        if ENRICH_CHAIN is None:
            payload, status = LM.enrich_simple(b.author, b.title, lang=lang)
            apply_enrichment_to_book(session, b, payload, status)
            (updated if status == "ok" else failed).append(b.id)  # type: ignore[arg-type]
        else:
            result = ENRICH_CHAIN.enrich(b.author, b.title, lang)
            status = "ok" if result.payload else "failed"
            apply_enrichment_to_book(session, b, result.payload, status)
            (updated if status == "ok" else failed).append(b.id)  # type: ignore[arg-type]
    return {"updated": updated, "failed": failed}


class EnrichAllRequest(BaseModel):
    only_pending: bool = Field(default=True, description="Procesar solo libros sin enrichment ok")
    only_never: bool = Field(default=False, description="Procesar únicamente libros nunca enriquecidos (status 'none')")
    throttle_ms: int = Field(default=0, ge=0, description="Pausa entre llamadas para no saturar el LM")
    limit: Optional[int] = Field(default=None, description="Limitar número de libros a procesar")
    language: Optional[str] = Field(default=None, description="Idioma preferido para el enriquecimiento")


@app.post("/enrich/all")
def enrich_all(req: EnrichAllRequest, background: BackgroundTasks, session: Session = Depends(get_session)):
    if ENRICH_PROGRESS.get("running"):
        return {"status": "already_running", "progress": ENRICH_PROGRESS}

    stmt = select(Book)
    if req.only_never:
        stmt = stmt.where(Book.enrichment_status == "none")
    elif req.only_pending:
        stmt = stmt.where(Book.enrichment_status != "ok")
    if req.limit is not None and req.limit > 0:
        stmt = stmt.limit(req.limit)
    # Precompute candidate ids to avoid holding ORM objects across threads
    ids = [b.id for b in session.exec(stmt).all() if b is not None]
    # Remove None ids for task typing safety
    ids = [cast(int, i) for i in ids if i is not None]

    def task(ids_: List[int], throttle_ms: int):
        ENRICH_PROGRESS.update({
            "running": True,
            "total": len(ids_),
            "processed": 0,
            "updated": 0,
            "failed": 0,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "last_book_id": None,
        })
        try:
            with Session(ENGINE) as s:  # type: ignore[arg-type]
                for bid in ids_:
                    b = s.get(Book, bid)
                    if not b:
                        ENRICH_PROGRESS["failed"] += 1
                        ENRICH_PROGRESS["processed"] += 1
                        continue
                    ENRICH_PROGRESS["last_book_id"] = bid
                    lang = req.language or CONFIG.default_language
                    if ENRICH_CHAIN is None:
                        payload, status = LM.enrich_simple(b.author, b.title, lang=lang)
                        apply_enrichment_to_book(s, b, payload, status)
                    else:
                        res = ENRICH_CHAIN.enrich(b.author, b.title, lang)
                        status = "ok" if res.payload else "failed"
                        apply_enrichment_to_book(s, b, res.payload, status)
                    if status == "ok":
                        ENRICH_PROGRESS["updated"] += 1
                    else:
                        ENRICH_PROGRESS["failed"] += 1
                    ENRICH_PROGRESS["processed"] += 1
                    if throttle_ms:
                        time.sleep(throttle_ms / 1000.0)
        finally:
            ENRICH_PROGRESS["running"] = False
            ENRICH_PROGRESS["finished_at"] = datetime.now(timezone.utc).isoformat()

    background.add_task(task, ids, req.throttle_ms)
    return {"status": "started", "candidates": len(ids)}


@app.get("/enrich/status")
def enrich_status() -> Dict[str, Any]:
    return ENRICH_PROGRESS


@app.post("/enrich/{book_id}")
def enrich_one(book_id: int, language: Optional[str] = Query(default=None), session: Session = Depends(get_session)):
    LOGGER.info("Enrich request for book ID: %d", book_id)
    book = session.get(Book, book_id)
    if not book:
        LOGGER.warning("Book not found: %d", book_id)
        raise HTTPException(status_code=404, detail="Book not found")
    
    LOGGER.info("Enriching book: '%s' by %s", book.title, book.author)
    lang = language or CONFIG.default_language
    if ENRICH_CHAIN is None:
        payload, status = LM.enrich_simple(book.author, book.title, lang=lang)
        book = apply_enrichment_to_book(session, book, payload, status)
        LOGGER.info("Enrich completed for book ID %d using LM (fallback): status=%s", book_id, status)
        return {"status": status, "enriched": payload}
    res = ENRICH_CHAIN.enrich(book.author, book.title, lang)
    status = "ok" if res.payload else "failed"
    book = apply_enrichment_to_book(session, book, res.payload, status)
    
    LOGGER.info("Enrich completed for book ID %d via chain: status=%s (source=%s, error=%s)", book_id, status, res.source, res.error)
    return {"status": status, "enriched": res.payload, "source": res.source, "error": res.error}


# ------------------------------------------------------------
# Dev entrypoint
# ------------------------------------------------------------


def _dev_main() -> None:
    import uvicorn

    LOGGER.info("The Archive starting with DB=%s, library=%s", DB_PATH, CONFIG.library_path)
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    _dev_main()

# Initialize on import for normal runs
init_app()
