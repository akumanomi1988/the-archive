"""
Skald — Minimal local system for indexing, searching, viewing, and downloading EPUB with optional local LLM enrichment (LMStudio).

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
from typing import Any, Dict, Iterable, List, Optional, Tuple
import threading
from contextlib import contextmanager

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
import hashlib


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------


class AppConfig(BaseModel):
    library_path: str = Field(default=str(Path.cwd() / "library"))
    db_path: str = Field(default=str(Path.cwd() / "skald.db"))
    lm_enabled: bool = False
    lm_url: str = "http://localhost:1234/api/predict"
    lm_timeout: float = 15.0
    lm_mock: bool = True
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])
    page_size_default: int = 20
    enrichment_batch_size: int = 10
    # Optional model name for OpenAI-style LMStudio endpoint
    lm_model: Optional[str] = "openai/gpt-oss-20"
    # LM rate limiting
    lm_max_concurrency: int = 5
    lm_min_interval_ms: int = 250

    @field_validator("page_size_default", "enrichment_batch_size", "lm_max_concurrency", "lm_min_interval_ms")
    def _positive(cls, v: int):  # type: ignore[override]
        if v <= 0:
            raise ValueError("must be positive")
        return v


def load_config(path_override: Optional[str | Path] = None) -> AppConfig:
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
                with p.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                return AppConfig(**data)
        except Exception as e:
            logging.getLogger("skald").warning("Failed loading config from %s: %s", p, e)
    logging.getLogger("skald").info("Using default in-memory config; create config.json or set SKALD_CONFIG to customize.")
    return AppConfig()


LOGGER = logging.getLogger("skald")
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
                    time.sleep(min_interval - delta)
            LM_LAST_REQUEST = time.monotonic()
        yield
    finally:
        sem.release()


def init_app(config_path: Optional[str] = None) -> None:
    """(Re)initialize config and database engine/tables.

    Safe to call multiple times in-process (used in tests).
    """
    global CONFIG, DB_PATH, ENGINE, LM, LM_SEMAPHORE, LM_LAST_REQUEST
    CONFIG = load_config(config_path)
    DB_PATH = Path(CONFIG.db_path)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    ENGINE = create_engine(f"sqlite:///{DB_PATH.as_posix()}", echo=False)
    SQLModel.metadata.create_all(ENGINE)
    # Init LM rate limiter
    LM_SEMAPHORE = threading.Semaphore(max(1, int(CONFIG.lm_max_concurrency)))
    LM_LAST_REQUEST = 0.0
    # Recreate LM client with possibly new config
    LM = LMClient(
        CONFIG.lm_url,
        timeout=CONFIG.lm_timeout,
        enabled=CONFIG.lm_enabled,
        mock=CONFIG.lm_mock,
        model=CONFIG.lm_model,
    )


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


def parse_epub_metadata(epub_path: Path) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    try:
        book = epub.read_epub(epub_path.as_posix())
        # title
        title_list = book.get_metadata("DC", "title")
        title = title_list[0][0] if title_list else None
        # author
        creators = book.get_metadata("DC", "creator")
        author = creators[0][0] if creators else None
        # language
        languages = book.get_metadata("DC", "language")
        language = languages[0][0] if languages else None
        return title, author, language
    except Exception as e:
        LOGGER.warning("Failed to parse EPUB metadata for %s: %s", epub_path, e)
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
    try:
        book = epub.read_epub(epub_path.as_posix())
        parts: List[str] = []
        count = 0
        for item in book.get_items():
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
        combined = "\n".join(parts)
        sanitized = bleach.clean(combined, tags=ALLOWED_TAGS, attributes=ALLOWED_ATTRS, strip=True)
        return sanitized
    except Exception as e:
        LOGGER.error("Failed to render EPUB %s: %s", epub_path, e)
        raise


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
        timeout: float = 15.0,
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

        tries = 10
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
        # Legacy endpoint should always be root + /api/predict
        legacy_url = root.rstrip("/") + "/api/predict"
        LOGGER.debug("LM endpoints openai=%s legacy=%s", openai_url, legacy_url)
        for attempt in range(1, tries + 1):
            try:
                # Try OpenAI-style first
                prompt = self._build_prompt(author, title)
                schema = self._json_schema()
                openai_payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": (
                            "Eres un experto bibliotecario. Tu tarea es completar metadatos de libros. "
                            "Responde siempre en JSON válido y exclusivamente JSON, sin texto adicional, sin markdown, sin bloques de código. "
                            "Si desconoces un campo, usa null o []. Respeta estrictamente el esquema y los tipos."
                        )},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 900,
                    # LMStudio requires json_schema or text
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "book_enrichment",
                            "schema": schema,
                        },
                    },
                }
                with httpx.Client(timeout=self.timeout) as client:
                    with lm_rate_limit():
                        resp = client.post(
                            openai_url,
                            json=openai_payload,
                            headers={"Content-Type": "application/json", "Accept": "application/json"},
                        )
                if resp.status_code == 200:
                    data = resp.json()
                    content = (data.get("choices", [{}])[0].get("message", {}).get("content", ""))
                    parsed = json.loads(content) if isinstance(content, str) else content
                    payload = self._validate_payload(parsed)
                    # Quick sanity check: ensure at least one meaningful field present
                    meaningful_keys = [
                        k for k in ("genre", "year", "series", "audience", "tags", "premise")
                        if payload.get(k)
                    ]
                    if not meaningful_keys:
                        raise ValueError("Empty/meaningless enrichment payload")
                    payload.setdefault("enriched_by", f"LMStudio:{self.model}")
                    payload.setdefault("enriched_at", now_iso)
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
            except Exception as e1:
                last_err = e1
                LOGGER.warning("LM enrich attempt %d (openai) failed: %s", attempt, e1)
                # Try legacy endpoint as fallback
                try:
                    with httpx.Client(timeout=self.timeout) as client:
                        with lm_rate_limit():
                            resp2 = client.post(
                                legacy_url,
                                json={"autor": author, "titulo": title},
                                headers={"Content-Type": "application/json", "Accept": "application/json"},
                            )
                    if resp2.status_code != 200:
                        raise RuntimeError(f"Legacy HTTP {resp2.status_code}")
                    # Some legacy endpoints return plain text; parse loosely
                    try:
                        data2 = resp2.json()
                    except Exception:
                        data2 = self._parse_json_loose(resp2.text)
                    payload2 = self._validate_payload(data2)
                    meaningful_keys2 = [
                        k for k in ("genre", "year", "series", "audience", "tags", "premise")
                        if payload2.get(k)
                    ]
                    if not meaningful_keys2:
                        raise ValueError("Empty/meaningless enrichment payload (legacy)")
                    payload2.setdefault("enriched_by", "LMStudio")
                    payload2.setdefault("enriched_at", now_iso)
                    return payload2, "ok"
                except Exception as e2:
                    last_err = e2
                    LOGGER.warning("LM enrich attempt %d (legacy) failed: %s", attempt, e2)
            # Exponential backoff with jitter between attempts
            if attempt < tries:
                backoff = min(0.5 * (2 ** (attempt - 1)), 8.0)  # cap at 8s
                time.sleep(backoff + random.random() * 0.2)
        LOGGER.error("LM enrich failed after retries: %s", last_err)
        return None, "failed"

    def _build_prompt(self, author: str, title: str) -> str:
        # Spanish prompt with embedded JSON template (if available) and clear constraints
        schema_or_template = self._template_json_str or (
            "{\n"
            '  "genre": null,\n'
            '  "year": null,\n'
            '  "series": null,\n'
            '  "series_number": null,\n'
            '  "audience": null,\n'
            '  "tags": [],\n'
            '  "content_warnings": [],\n'
            '  "premise": null,\n'
            '  "confidence": null,\n'
            '  "localized": {\n'
            '    "es": {"premise": null, "tags": []},\n'
            '    "en": {"premise": null, "tags": []},\n'
            '    "fr": {"premise": null, "tags": []},\n'
            '    "de": {"premise": null, "tags": []},\n'
            '    "pt": {"premise": null, "tags": []}\n'
            "  }\n"
            "}"
        )
        return (
            "Eres un experto bibliotecario y analista literario. "
            "Necesito que completes el siguiente JSON de metadatos para el libro indicado.\n\n"
            f"Libro: autor=\"{author}\", titulo=\"{title}\".\n\n"
            "Instrucciones estrictas:\n"
            "- Devuelve ÚNICAMENTE un JSON válido. Nada de explicaciones, ni markdown, ni bloques ```json.\n"
            "- Rellena valores plausibles; si un dato no se conoce, usa null o [].\n"
            "- Respeta los nombres de clave y tipos. No añadas claves nuevas.\n"
            "- Las etiquetas (tags) deben ser cortas (1–3 palabras).\n"
            "- Incluye al menos un campo informativo (género, tags o premisa).\n"
            "- Si puedes, completa también localized.es con la sinopsis y etiquetas en español.\n\n"
            "Plantilla a rellenar exactamente con estas claves:\n"
            f"{schema_or_template}\n"
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
        """JSON Schema for enforced structured output."""
        lang_block = {
            "type": "object",
            "properties": {
                "premise": {"type": ["string", "null"]},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "additionalProperties": False,
        }
        schema: Dict[str, Any] = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "genre": {"type": ["string", "null"]},
                "year": {"type": ["integer", "null"]},
                "series": {"type": ["string", "null"]},
                "series_number": {"type": ["integer", "null"]},
                "audience": {"type": ["string", "null"]},
                "tags": {"type": "array", "items": {"type": "string"}},
                "content_warnings": {"type": "array", "items": {"type": "string"}},
                "premise": {"type": ["string", "null"]},
                "confidence": {"type": ["number", "null"]},
                "localized": {
                    "type": ["object", "null"],
                    "properties": {
                        "es": lang_block,
                        "en": lang_block,
                        "fr": lang_block,
                        "de": lang_block,
                        "pt": lang_block,
                    },
                    "additionalProperties": False,
                },
            },
            "additionalProperties": False,
        }
        return schema

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
        year_val = None
        if payload.get("year") is not None:
            try:
                year_val = int(payload.get("year"))
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


app = FastAPI(title="Skald", version="0.1.0")

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
    return PlainTextResponse("Skald API running. Open /docs for API docs.")


@app.get("/filters")
def get_filters(session: Session = Depends(get_session)) -> Dict[str, Any]:
    # Authors
    authors_rows = session.exec(select(Book.author).distinct().order_by(Book.author)).all()
    authors: List[str] = [a for (a,) in authors_rows if isinstance(a, str)] if authors_rows and isinstance(authors_rows[0], tuple) else [a for a in authors_rows if isinstance(a, str)]
    # Genres (non-null)
    genres_rows = session.exec(select(Book.genre).where(Book.genre.is_not(None)).distinct().order_by(Book.genre)).all()
    genres: List[str] = [g for (g,) in genres_rows if isinstance(g, str)] if genres_rows and isinstance(genres_rows[0], tuple) else [g for g in genres_rows if isinstance(g, str)]
    # Years
    min_year = session.exec(select(Book.year).where(Book.year.is_not(None)).order_by(Book.year)).first()
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
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=CONFIG.page_size_default, ge=1, le=200),
    session: Session = Depends(get_session),
) -> PaginatedBooks:
    state_filter: Optional[str] = Query(default=None, description="Filter by state: favorite|read|pending")
    stmt = select(Book)
    if q:
        like = f"%{q}%"
        stmt = stmt.where((Book.title.ilike(like)) | (Book.author.ilike(like)))
    if autor:
        stmt = stmt.where(Book.author.ilike(f"%{autor}%"))
    if titulo:
        stmt = stmt.where(Book.title.ilike(f"%{titulo}%"))
    if genre:
        stmt = stmt.where(Book.genre.ilike(f"%{genre}%"))
    if year_from is not None:
        stmt = stmt.where((Book.year.is_not(None)) & (Book.year >= year_from))
    if year_to is not None:
        stmt = stmt.where((Book.year.is_not(None)) & (Book.year <= year_to))

    # State filtering if provided
    # We fetch after building the text filters to reduce set size
    # To keep SQL simple across SQLite versions, use IN (subquery)
    # Validate state_filter
    if state_filter in {"favorite", "read", "pending"}:
        flag = {
            "favorite": BookState.favorite_flag,
            "read": BookState.read_flag,
            "pending": BookState.pending_flag,
        }[state_filter]
        sub = select(BookState.book_id).where(flag == True)  # noqa: E712
        stmt = stmt.where(Book.id.in_(sub))

    total = len(session.exec(stmt).all())
    # pagination
    offset = (page - 1) * page_size
    stmt = stmt.offset(offset).limit(page_size)
    rows = session.exec(stmt).all()
    ids = [b.id for b in rows if b is not None]
    # Map states for these ids
    state_map: Dict[int, BookState] = {}
    if ids:
        st_rows = session.exec(select(BookState).where(BookState.book_id.in_(ids))).all()
        for st in st_rows:
            if st and st.book_id:
                state_map[st.book_id] = st
    items = []
    for b in rows:
        if not b:
            continue
        st = state_map.get(b.id or -1)
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
    }


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


@app.post("/enrich/batch")
def enrich_batch(req: EnrichBatchRequest, session: Session = Depends(get_session)):
    ids = req.ids
    updated: List[int] = []
    failed: List[int] = []
    limit = CONFIG.enrichment_batch_size
    books: List[Book]
    if ids:
        books = [b for b in session.exec(select(Book).where(Book.id.in_(ids))).all() if b is not None]
    else:
        books = session.exec(
            select(Book).where(Book.enrichment_status != "ok").limit(limit)
        ).all()
    for b in books:
        payload, status = LM.enrich(b.author, b.title)
        apply_enrichment_to_book(session, b, payload, status)
        (updated if status == "ok" else failed).append(b.id)  # type: ignore[arg-type]
    return {"updated": updated, "failed": failed}


class EnrichAllRequest(BaseModel):
    only_pending: bool = Field(default=True, description="Procesar solo libros sin enrichment ok")
    only_never: bool = Field(default=False, description="Procesar únicamente libros nunca enriquecidos (status 'none')")
    throttle_ms: int = Field(default=0, ge=0, description="Pausa entre llamadas para no saturar el LM")
    limit: Optional[int] = Field(default=None, description="Limitar número de libros a procesar")


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
                    payload, status = LM.enrich(b.author, b.title)
                    apply_enrichment_to_book(s, b, payload, status)
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
def enrich_one(book_id: int, session: Session = Depends(get_session)):
    book = session.get(Book, book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    payload, status = LM.enrich(book.author, book.title)
    book = apply_enrichment_to_book(session, book, payload, status)
    return {"status": status, "enriched": payload}


# ------------------------------------------------------------
# Dev entrypoint
# ------------------------------------------------------------


def _dev_main() -> None:
    import uvicorn

    LOGGER.info("Skald starting with DB=%s, library=%s", DB_PATH, CONFIG.library_path)
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    _dev_main()

# Initialize on import for normal runs
init_app()
