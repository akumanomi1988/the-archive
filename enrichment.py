from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

try:
    # Optional dependency; provider will gracefully degrade if missing
    from g4f.client import Client as G4FClient  # type: ignore
    _G4F_AVAILABLE = True
except Exception:
    _G4F_AVAILABLE = False


# ---------------------------
# Data structures
# ---------------------------

@dataclass
class EnrichmentResult:
    payload: Optional[Dict[str, Any]]
    source: str
    error: Optional[str] = None


class EnrichmentProvider:
    name = "base"
    timeout: float = 10.0

    def enrich(self, author: str, title: str, lang: str) -> EnrichmentResult:
        return EnrichmentResult(None, self.name, "not implemented")


# ---------------------------
# Utilities (clean/parse/normalize)
# ---------------------------

def clean_text(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    t = unicodedata.normalize("NFKC", str(text))
    replacements = {
        "\u2011": "-",
        "\u2013": "-",
        "\u2014": "--",
        "\u2018": "'",
        "\u2019": "'",
        "\u201C": '"',
        "\u201D": '"',
        "\u2026": "...",
        "\u00A0": " ",
    }
    for k, v in replacements.items():
        t = t.replace(k, v)
    t = ''.join(ch for ch in t if unicodedata.category(ch)[0] not in 'C' or ch in '\t\n\r ')
    t = ' '.join(t.split())
    return t


def parse_json_loose(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    matches = re.findall(r"\{[\s\S]*\}", text)
    if not matches:
        return None
    last = matches[-1]
    try:
        return json.loads(last)
    except Exception:
        return None


def normalize_payload(p: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "genre": p.get("genre"),
        "year": None,
        "series": p.get("series") if isinstance(p.get("series"), str) else None,
        "series_number": None,
        "audience": p.get("audience") if p.get("audience") in {"children", "young_adult", "adult", "general"} else "general",
        "tags": [str(x) for x in (p.get("tags") or [])][:8],
        "content_warnings": [str(x) for x in (p.get("content_warnings") or [])][:3],
        "premise": clean_text(p.get("premise")),
        "confidence": None,
        "enriched_by": p.get("enriched_by"),
        "enriched_at": p.get("enriched_at"),
    # Optional extended fields
    "rating": None,
    "ratings_count": None,
    "pages": None,
    "publisher": p.get("publisher") if isinstance(p.get("publisher"), str) else None,
    }
    y = p.get("year")
    try:
        out["year"] = int(y) if y is not None else None
    except Exception:
        out["year"] = None
    c = p.get("confidence")
    try:
        out["confidence"] = float(c) if c is not None else None
    except Exception:
        out["confidence"] = None
    # ratings
    r = p.get("rating")
    try:
        out["rating"] = float(r) if r is not None else None
    except Exception:
        out["rating"] = None
    rc = p.get("ratings_count")
    try:
        out["ratings_count"] = int(rc) if rc is not None else None
    except Exception:
        out["ratings_count"] = None
    pg = p.get("pages")
    try:
        out["pages"] = int(pg) if pg is not None else None
    except Exception:
        out["pages"] = None
    if isinstance(out["genre"], str) and len(out["genre"]) > 60:
        out["genre"] = out["genre"].split(" / ")[0][:60]
    return out


# ---------------------------
# Providers
# ---------------------------

class OpenLibraryProvider(EnrichmentProvider):
    name = "openlibrary"

    def __init__(self, timeout: float = 8.0) -> None:
        self.timeout = timeout

    def _pick_best(self, docs: List[Dict[str, Any]], author: str, title: str) -> Optional[Dict[str, Any]]:
        # Simple heuristic: prefer exact/near title match and author contains
        t_norm = (title or "").casefold().strip()
        a_norm = (author or "").casefold().strip()
        best: Optional[Dict[str, Any]] = None
        best_score = -1
        for d in docs:
            score = 0
            dt = (d.get("title") or "").casefold().strip()
            if dt == t_norm:
                score += 3
            elif t_norm and t_norm in dt:
                score += 1
            auths = [str(x) for x in (d.get("author_name") or [])]
            if any(a_norm and a_norm in str(a).casefold() for a in auths):
                score += 2
            year = d.get("first_publish_year")
            if isinstance(year, int):
                score += 1
            if score > best_score:
                best = d
                best_score = score
        return best

    def enrich(self, author: str, title: str, lang: str) -> EnrichmentResult:
        try:
            params = {"title": title, "author": author, "limit": 5}
            with httpx.Client(timeout=self.timeout) as client:
                r = client.get("https://openlibrary.org/search.json", params=params)
                if r.status_code != 200:
                    return EnrichmentResult(None, self.name, f"http {r.status_code}")
                data = r.json()
            docs = data.get("docs") or []
            if not docs:
                return EnrichmentResult(None, self.name, "no results")
            d = self._pick_best(docs, author, title) or docs[0]
            year = d.get("first_publish_year")
            subjects = d.get("subject") or []
            tags = [str(s) for s in subjects[:8]]
            premise = None
            rating = None
            ratings_count = None
            pages = None
            publisher = None
            if d.get("key"):
                work_key = d["key"]
                try:
                    with httpx.Client(timeout=self.timeout) as client:
                        w = client.get(f"https://openlibrary.org{work_key}.json")
                        if w.status_code == 200:
                            wj = w.json()
                            desc = wj.get("description")
                            if isinstance(desc, dict):
                                premise = desc.get("value")
                            elif isinstance(desc, str):
                                premise = desc
                            pubs = wj.get("publishers") or []
                            if isinstance(pubs, list) and pubs:
                                publisher = pubs[0] if isinstance(pubs[0], str) else None
                    # Ratings endpoint
                    with httpx.Client(timeout=self.timeout) as client:
                        wr = client.get(f"https://openlibrary.org{work_key}/ratings.json")
                        if wr.status_code == 200:
                            rj = wr.json() or {}
                            summ = rj.get("summary") or {}
                            rating = summ.get("average")
                            ratings_count = summ.get("count")
                except Exception:
                    pass
            # Try pages from first edition if available
            eds = d.get("edition_key") or []
            if isinstance(eds, list) and eds:
                ek = eds[0]
                try:
                    with httpx.Client(timeout=self.timeout) as client:
                        e = client.get(f"https://openlibrary.org/books/{ek}.json")
                        if e.status_code == 200:
                            ej = e.json()
                            np = ej.get("number_of_pages")
                            if isinstance(np, int):
                                pages = np
                            if not publisher and isinstance(ej.get("publishers"), list) and ej.get("publishers"):
                                pub0 = ej.get("publishers")[0]
                                publisher = pub0 if isinstance(pub0, str) else publisher
                except Exception:
                    pass
            payload = {
                "genre": (d.get("subject_facet") or tags or [None])[0],
                "year": int(year) if isinstance(year, int) else None,
                "series": None,
                "series_number": None,
                "audience": "general",
                "tags": tags,
                "content_warnings": [],
                "premise": clean_text(premise) if premise else None,
                "confidence": 0.6,
                "enriched_by": "OpenLibrary",
                "enriched_at": datetime.now(timezone.utc).isoformat(),
                "rating": rating,
                "ratings_count": ratings_count,
                "pages": pages,
                "publisher": publisher,
            }
            return EnrichmentResult(normalize_payload(payload), self.name, None)
        except Exception as e:
            return EnrichmentResult(None, self.name, str(e))


class GoogleBooksProvider(EnrichmentProvider):
    name = "googlebooks"

    def __init__(self, timeout: float = 8.0) -> None:
        self.timeout = timeout

    def enrich(self, author: str, title: str, lang: str) -> EnrichmentResult:
        try:
            with httpx.Client(timeout=self.timeout) as client:
                r = client.get(
                    "https://www.googleapis.com/books/v1/volumes",
                    params={"q": f'intitle:"{title}"+inauthor:"{author}"', "maxResults": 5},
                )
            if r.status_code != 200:
                return EnrichmentResult(None, self.name, f"http {r.status_code}")
            data = r.json()
            items = data.get("items") or []
            if not items:
                return EnrichmentResult(None, self.name, "no results")
            info = items[0].get("volumeInfo", {})
            published = (info.get("publishedDate") or "")[:4]
            try:
                year = int(published) if published.isdigit() else None
            except Exception:
                year = None
            tags = [str(x) for x in (info.get("categories") or [])][:8]
            rating = info.get("averageRating")
            ratings_count = info.get("ratingsCount")
            pages = info.get("pageCount")
            publisher = info.get("publisher") if isinstance(info.get("publisher"), str) else None
            payload = {
                "genre": tags[0] if tags else None,
                "year": year,
                "series": None,
                "series_number": None,
                "audience": "general",
                "tags": tags,
                "content_warnings": [],
                "premise": clean_text(info.get("description")) if info.get("description") else None,
                "confidence": 0.55,
                "enriched_by": "GoogleBooks",
                "enriched_at": datetime.now(timezone.utc).isoformat(),
                "rating": rating,
                "ratings_count": ratings_count,
                "pages": pages,
                "publisher": publisher,
            }
            return EnrichmentResult(normalize_payload(payload), self.name, None)
        except Exception as e:
            return EnrichmentResult(None, self.name, str(e))


class G4FProvider(EnrichmentProvider):
    name = "g4f"

    def __init__(self, model: Optional[str] = None, timeout: float = 30.0) -> None:
        self.model = model or "gpt-4o-mini"
        self.timeout = timeout
        self._available = _G4F_AVAILABLE

    def enrich(self, author: str, title: str, lang: str) -> EnrichmentResult:
        if not self._available:
            return EnrichmentResult(None, self.name, "g4f not installed")
        try:
            client = G4FClient()  # type: ignore
            system = (
                "You return ONLY a JSON object with fields: genre, year, audience (children|young_adult|adult|general), "
                "confidence (0-1), tags (max 8), premise (100-180 words, spoiler-free)."
            )
            user = f'Language: "{lang}". Author: "{author}". Title: "{title}". Return only JSON.'
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                timeout=self.timeout,
            )
            content = resp.choices[0].message.content.strip()
            payload = parse_json_loose(content)
            if not payload:
                return EnrichmentResult(None, self.name, "empty")
            payload["enriched_by"] = f"g4f:{self.model}"
            payload["enriched_at"] = datetime.now(timezone.utc).isoformat()
            return EnrichmentResult(normalize_payload(payload), self.name, None)
        except Exception as e:
            return EnrichmentResult(None, self.name, str(e))


class LMStudioProvider(EnrichmentProvider):
    name = "lmstudio"

    def __init__(self, base_url: str, model: Optional[str], timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def enrich(self, author: str, title: str, lang: str) -> EnrichmentResult:
        try:
            url = self.base_url
            if not url.endswith("/v1/chat/completions"):
                url = url + "/v1/chat/completions"
            prompt = (
                "Analyze this book and return only a JSON object with keys: "
                'genre (string|null), year (int|null), audience ("children"|"young_adult"|"adult"|"general"), '
                'confidence (0.0-1.0), tags (array of strings, max 8), premise (string|null, 120-200 words, spoiler-free). '
                f'Write ALL fields in this language: "{lang}". '
                f'Author: "{author}". Title: "{title}".'
            )
            with httpx.Client(timeout=self.timeout) as client:
                r = client.post(
                    url,
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": "Return only JSON."},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.2,
                        "max_tokens": 800,
                    },
                    headers={"Content-Type": "application/json"},
                )
            if r.status_code != 200:
                return EnrichmentResult(None, self.name, f"http {r.status_code}: {r.text[:200]}")
            content = (
                (r.json().get("choices", [{}])[0].get("message", {}) or {}).get("content", "").strip()
            )
            payload = parse_json_loose(content)
            if not payload:
                return EnrichmentResult(None, self.name, "empty")
            payload["enriched_by"] = f"LMStudio:{self.model}"
            payload["enriched_at"] = datetime.now(timezone.utc).isoformat()
            return EnrichmentResult(normalize_payload(payload), self.name, None)
        except Exception as e:
            return EnrichmentResult(None, self.name, str(e))


# ---------------------------
# Chain
# ---------------------------

class ChainEnricher:
    def __init__(self, providers: List[EnrichmentProvider]) -> None:
        self.providers = providers
        # simple in-memory cache: key=(author|title|lang)
        self._cache: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _cache_key(author: str, title: str, lang: str) -> str:
        return f"{(author or '').strip().casefold()}|{(title or '').strip().casefold()}|{(lang or '').strip().casefold()}"

    @staticmethod
    def _merge(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
        # fill only missing/empty fields in base with extra
        out = dict(base)
        for k, v in extra.items():
            if k not in out or out[k] in (None, "", [], {}):
                out[k] = v
        return out

    def enrich(self, author: str, title: str, lang: str) -> EnrichmentResult:
        last_err: Optional[str] = None
        key = self._cache_key(author, title, lang)
        accumulated: Optional[Dict[str, Any]] = self._cache.get(key)

        for p in self.providers:
            res = p.enrich(author, title, lang)
            if res.payload:
                if accumulated is None:
                    accumulated = res.payload
                else:
                    # incremental: complete missing fields
                    accumulated = self._merge(accumulated, res.payload)
                # criterio de suficiencia: tener al menos genre o >=3 tags o premise >= 80 chars
                genre_ok = bool((accumulated.get("genre") or "").strip())
                tags_ok = isinstance(accumulated.get("tags"), list) and len(accumulated.get("tags") or []) >= 3
                premise_len = len((accumulated.get("premise") or ""))
                if genre_ok or tags_ok or premise_len >= 80:
                    # guarda en caché normalizado
                    self._cache[key] = accumulated
                    return EnrichmentResult(accumulated, p.name, None)
            last_err = res.error

        if accumulated:
            # aunque no cumpla el umbral, devuelve lo mejor que hay y cachea
            self._cache[key] = accumulated
            return EnrichmentResult(accumulated, "chain-partial", last_err)
        return EnrichmentResult(None, "chain", last_err or "all providers failed")
