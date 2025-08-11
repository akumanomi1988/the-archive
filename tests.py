import json
from pathlib import Path

import pytest
import pytest_asyncio
import httpx

import backend


@pytest_asyncio.fixture(scope="session")
async def client(tmp_path_factory):
    # Use a temp DB and library for isolation
    tmp_dir = tmp_path_factory.mktemp("skald")
    cfg = {
        "library_path": str(tmp_dir / "library"),
        "db_path": str(tmp_dir / "skald.db"),
        "lm_enabled": True,
        "lm_mock": True,
        "cors_origins": ["*"],
    }
    # Write a temporary config and reload
    cfg_path = tmp_dir / "config.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    # Patch env and re-init app
    import os
    os.environ["SKALD_CONFIG"] = str(cfg_path)
    backend.init_app()

    # Create fake EPUB structure
    lib = Path(backend.CONFIG.library_path)
    author_dir = lib / "Test_Author"
    author_dir.mkdir(parents=True, exist_ok=True)
    # Create a minimal EPUB file
    epub_file = author_dir / "Test_Title.epub"
    create_min_epub(epub_file, title="Test Title", author="Test Author")

    transport = httpx.ASGITransport(app=backend.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c


def create_min_epub(path: Path, title: str, author: str):
    from ebooklib import epub

    book = epub.EpubBook()
    book.set_title(title)
    book.add_author(author)
    c1 = epub.EpubHtml(title="Intro", file_name="chap_01.xhtml", lang="en")
    c1.set_content("<h1>Intro</h1><p>Hello</p>")
    book.add_item(c1)
    book.toc = (epub.Link("chap_01.xhtml", "Intro", "intro"),)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", c1]
    epub.write_epub(str(path), book)


@pytest.mark.asyncio
async def test_health(client: httpx.AsyncClient):
    r = await client.get("/health")
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_reindex_and_search(client: httpx.AsyncClient):
    r = await client.post("/reindex", json={"mode": "sync"})
    assert r.status_code == 200
    data = r.json()
    assert data["added"] >= 1

    r = await client.get("/books", params={"q": "Test", "page": 1, "page_size": 10})
    assert r.status_code == 200
    lst = r.json()
    assert lst["total"] >= 1
    assert len(lst["items"]) >= 1


@pytest.mark.asyncio
async def test_open_and_download(client: httpx.AsyncClient):
    # find one id
    r = await client.get("/books", params={"page": 1, "page_size": 1})
    b = r.json()["items"][0]
    book_id = b["id"]

    r = await client.get(f"/open/{book_id}")
    assert r.status_code == 200
    assert "<html" not in r.text.lower()  # returns body content only

    r = await client.get(f"/download/{book_id}")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/epub+zip")


@pytest.mark.asyncio
async def test_enrich(client: httpx.AsyncClient):
    # find one id
    r = await client.get("/books", params={"page": 1, "page_size": 1})
    book_id = r.json()["items"][0]["id"]

    r = await client.post(f"/enrich/{book_id}")
    assert r.status_code == 200
    payload = r.json()
    assert payload["status"] in {"ok", "failed"}

    r = await client.post("/enrich/batch", json={"ids": [book_id]})
    assert r.status_code == 200
