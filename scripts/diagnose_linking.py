"""Diagnostic checks for Book/Author/BookAuthor counts and samples.
Run from repo root with the project's venv active.
"""
from __future__ import annotations
import sys
import pathlib
from sqlalchemy import func
from sqlmodel import select

root = pathlib.Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import backend

backend.init_app()
with backend.Session(backend.ENGINE) as session:
    total_books = int(session.exec(select(func.count()).select_from(backend.Book)).one())
    total_authors = int(session.exec(select(func.count()).select_from(backend.Author)).one())
    total_bookauthors = int(session.exec(select(func.count()).select_from(backend.BookAuthor)).one())

    print(f"total_books={total_books}")
    print(f"total_authors={total_authors}")
    print(f"total_bookauthors={total_bookauthors}")

    # sample books without BookAuthor
    q = select(backend.Book).where(backend.Book.id.notin_(select(backend.BookAuthor.book_id)))
    rows = session.exec(q.limit(5)).all()
    print("\nSample books without BookAuthor:")
    for b in rows:
        print(f"id={b.id} title={b.title!r} author={b.author!r}")

    # sample authors
    authors = session.exec(select(backend.Author).limit(5)).all()
    print("\nSample authors:")
    for a in authors:
        print(f"id={a.id} name={a.name!r}")

    # approximate strict join matches
    joincond = func.lower(func.trim(backend.Book.author)) == func.lower(func.trim(backend.Author.name))
    q2 = select(func.count()).select_from(backend.Book).join(backend.Author, joincond).where(backend.Book.id.notin_(select(backend.BookAuthor.book_id)))
    cnt_strict = int(session.exec(q2).one())
    print("\napprox strict join matches (raw SQL-level):", cnt_strict)

print("diagnostic done")
