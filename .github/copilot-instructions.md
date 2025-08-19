Repository: The Archive — EPUB indexer, viewer, and optional LLM enrichment (LMStudio)

Purpose of this file
- Give a new coding agent clear, high-value guidance to make safe, fast, and testable code changes in this repository.
- Avoid wasted searches and reduce the likelihood of introducing failing builds or runtime regressions.

Quick summary
- Single-file FastAPI backend (`backend.py`) implementing: config, DB models (SQLModel/SQLite), indexer, enrichment integration, EPUB parsing, cover extraction/caching, and REST endpoints.
- Small SPA frontend: `index.html` (no build step). Tests in `tests.py` use pytest + httpx.
- Core languages/tools: Python 3.11+, FastAPI, SQLModel/SQLAlchemy, ebooklib, httpx, Pillow (thumbnailing), bleach, BeautifulSoup, lxml.

Constraints for agents
- Keep changes minimal and well-scoped. This repo contains a single large backend file; prefer small, localized edits or logical refactors (extract functions to new modules) and run tests.
- Always preserve runtime compatibility with Python 3.11.

Build / dev / test commands (validated)
- Create a virtualenv and install pinned dependencies (always do this first):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

- Run tests (quick smoke):

```powershell
pytest -q
```

- Run server (one of):

```powershell
# Simple run
python backend.py

# Production-ish
uvicorn backend:app --host 0.0.0.0 --port 8000
```

Notes & pitfalls discovered while validating commands
- Python 3.11 is required. Using lower versions may cause SQLModel/Pydantic or typing issues.
- Tests create temporary config files via SKALD_CONFIG; ensure `pytest` runs in a clean environment and that `ebooklib` can write temp files.
- Pillow is required for thumbnail generation; if not present, thumbnail functions are written to degrade gracefully. Still, pinning Pillow in `requirements.txt` is recommended (already present).
- If tests fail due to missing system libs for `lxml` or `Pillow` on CI, document the system dependency (e.g., libxml2/libxslt or Windows build tools) and add to README or CI setup.

Project layout (high-level)
- `backend.py` — main application and logic (single file). Edit with care. Key symbol locations:
  - DB models: `class Book`, `class BookState`.
  - Cover helpers: functions starting `_cover_cache_path_for_sha`, `get_or_build_cover`, `try_extract_cover_from_epub`.
  - Enrichment integration: imports from `enrichment.py` and `ENRICH_CHAIN` usage.
  - Endpoints: `list_books`, `get_book`, `get_cover`, `get_cover_thumb`, `open_book`, `download_book`, `/enrich` endpoints.
- `enrichment.py` — provider chain for OpenLibrary, Google Books, g4f, LMStudio. This is safe to extend for new providers; uses httpx and has its own timeout defaults.
- `index.html` — UI; no build tool (static file served by backend).
- `requirements.txt` — pinned Python deps used by CI/dev.
- `tests.py` — pytest-based tests that spin up the ASGI app using httpx.ASGITransport; maintain them when changing API shapes.
- `config.json.example` — example runtime config. Tests override by writing a temp config and setting SKALD_CONFIG.

CI / Validation
- No GitHub Actions workflows are present in this repo. The primary validation is `pytest` and local run smoke tests.
- Validation checklist for PRs:
  1. Run `pip install -r requirements.txt` in a fresh venv.
  2. Run `pytest -q` and fix failures.
  3. Run the server with `uvicorn backend:app` and manually test key endpoints: `/health`, `/reindex` (sync), `/books`, `/open/{id}`, `/download/{id}`.
  4. If adding/renaming endpoints, update `tests.py` accordingly.

Editing guidelines for safe PRs
- Small diffs: Keep changes limited to a few functions/classes per PR. Large refactors should be split into multiple PRs with tests passing after each.
- API stability: If a change alters an endpoint response (keys/names), update `tests.py` before running tests.
- Typing & imports: `backend.py` uses typed SQLModel models; prefer adding new imports at the top and run a quick `pytest` to catch unresolved names.
- External calls: Providers use external HTTP APIs (OpenLibrary, Google Books). Mock or run tests with `lm_mock` or isolated network when needed.

Common searches to avoid repeating
- Where endpoints live: search for `@app.get("/books"` or `def list_books`.
- Where cover logic lives: search for `get_or_build_cover` and `_cover_cache_path_for_sha`.
- Enrichment providers: open `enrichment.py` (top-level classes named `OpenLibraryProvider`, `GoogleBooksProvider`, `G4FProvider`, `LMStudioProvider`).
- Tests: `tests.py` covers integration-level tests and shows how SKALD_CONFIG is used.

Fallback behavior and when to search
- Trust this file: follow the instructions here first. Only run repo-wide searches if:
  - The guidance in this file conflicts with observed code (e.g., an endpoint name changed).
  - You need to find a symbol not mentioned above.

Quick checklist for agents before creating a PR
- [ ] Run `pip install -r requirements.txt` in a fresh venv.
- [ ] Run `pytest -q` and ensure all tests pass locally.
- [ ] If tests require network or external services, set `lm_mock` in a temp config and use `SKALD_CONFIG` in tests.
- [ ] Start the app and smoke-test `/books`, `/open/{id}`, `/download/{id}`, `/enrich/{id}`.
- [ ] Keep PRs small and include updated tests for behavior changes.

If you still need to search
- Preferred quick searches:
  - `grep -n "def list_books" -R`
  - `grep -n "get_or_build_cover" -R`
  - `grep -n "ENRICH_CHAIN" -R`
  - `grep -n "pytest" -R`

End of instructions — trust these steps. Only search the repo when the above information proves incomplete or inconsistent with the codebase.
