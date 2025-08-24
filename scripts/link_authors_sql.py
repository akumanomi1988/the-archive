"""Link Book rows to Author rows using a single SQL statement.

Usage:
  python scripts/link_authors_sql.py --mode strict|substr [--dry-run]

- mode strict: matches where lower(trim(Book.author)) == lower(trim(Author.name))
- mode substr: matches where lower(Book.author) LIKE '%'||lower(Author.name)||'%' (broader)

This script is idempotent: it will only insert BookAuthor rows for books that don't already have one.
"""
from __future__ import annotations

import argparse
import logging
import sys
import pathlib
from typing import Optional

# Ensure project root is on sys.path when run as a script
root = pathlib.Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import backend
from sqlalchemy import text, func
from sqlmodel import select

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
LOG = logging.getLogger('link_authors')


def count_matches(mode: str) -> int:
    backend.init_app()
    session = backend.Session(backend.ENGINE)
    try:
        if mode == 'strict':
            q = text("""
            SELECT COUNT(*) FROM Book b
            JOIN Author a ON lower(trim(b.author)) = lower(trim(a.name))
            WHERE b.id NOT IN (SELECT book_id FROM BookAuthor)
            """)
        else:
            q = text("""
            SELECT COUNT(*) FROM Book b
            JOIN Author a ON lower(b.author) LIKE '%' || lower(a.name) || '%'
            WHERE b.id NOT IN (SELECT book_id FROM BookAuthor)
            """)
        res = session.exec(q).one()
        # session.exec(text) may return a Row; extract first
        cnt = int(res[0] if isinstance(res, (list, tuple)) or hasattr(res, '__len__') else res)
        return cnt
    finally:
        session.close()


def run_insert(mode: str) -> dict:
    backend.init_app()
    con = backend.ENGINE.connect()
    try:
        if mode == 'strict':
            sql = text('''
            INSERT INTO BookAuthor (book_id, author_id)
            SELECT b.id, a.id FROM Book b
            JOIN Author a ON lower(trim(b.author)) = lower(trim(a.name))
            WHERE b.id NOT IN (SELECT book_id FROM BookAuthor)
            ''')
        else:
            sql = text('''
            INSERT INTO BookAuthor (book_id, author_id)
            SELECT b.id, a.id FROM Book b
            JOIN Author a ON lower(b.author) LIKE '%' || lower(a.name) || '%'
            WHERE b.id NOT IN (SELECT book_id FROM BookAuthor)
            ''')
        con.execute(sql)
        con.commit()
        session = backend.Session(backend.ENGINE)
        try:
            # get scalar count properly
            total = int(session.exec(select(func.count()).select_from(backend.BookAuthor)).one())
        finally:
            session.close()
        return {"relations_total": int(total)}
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description='Link Book -> Author using SQL')
    parser.add_argument('--mode', choices=['strict', 'substr'], default='strict')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)

    LOG.info('Mode=%s dry_run=%s', args.mode, args.dry_run)
    cnt = count_matches(args.mode)
    LOG.info('Matches to link (approx): %d', cnt)
    if args.dry_run:
        return 0
    res = run_insert(args.mode)
    LOG.info('Inserted relations total now: %s', res.get('relations_total'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
