"""Backfill master tables from existing Book rows.

Usage:
  python scripts/backfill_from_books.py [--dry-run] [--batch-size N]

This script is idempotent: it will not duplicate existing relations.
It uses the helpers defined in `backend.py` and the DB engine there.
"""
from __future__ import annotations

import argparse
import logging
from typing import Optional

# Ensure we can import the app module
import backend
from sqlmodel import select

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
LOG = logging.getLogger('backfill')


def backfill_all(batch_size: int = 200, dry_run: bool = False) -> dict:
    backend.init_app()  # ensure engine/tables initialized
    engine = backend.ENGINE
    if engine is None:
        raise RuntimeError('Database engine not initialized')

    added_relations = 0
    processed = 0
    # Use a session directly
    with backend.Session(engine) as session:
        books = session.exec(select(backend.Book)).all()
        LOG.info('Found %d books to process', len(books))
        for b in books:
            processed += 1
            try:
                # Authors
                aid = backend.get_or_create_author_id(session, b.author)
                if aid:
                    exists = session.exec(select(backend.BookAuthor).where(backend.BookAuthor.book_id == b.id)).first()
                    if not exists:
                        LOG.debug('Would add BookAuthor for book %s -> author_id=%s', b.id, aid)
                        if not dry_run:
                            session.add(backend.BookAuthor(book_id=b.id, author_id=aid))
                            added_relations += 1
                # Genres
                gid = backend.get_or_create_genre_id(session, b.genre)
                if gid:
                    exists = session.exec(select(backend.BookGenre).where(backend.BookGenre.book_id == b.id)).first()
                    if not exists:
                        LOG.debug('Would add BookGenre for book %s -> genre_id=%s', b.id, gid)
                        if not dry_run:
                            session.add(backend.BookGenre(book_id=b.id, genre_id=gid))
                            added_relations += 1
            except Exception:
                LOG.exception('Failed processing book id=%s', b.id)
                session.rollback()

            # Commit in batches to avoid long transactions
            if not dry_run and (processed % batch_size == 0):
                try:
                    session.commit()
                    LOG.info('Committed batch, processed=%d', processed)
                except Exception:
                    LOG.exception('Commit failed at processed=%d', processed)
                    session.rollback()

        # final commit
        if not dry_run:
            try:
                session.commit()
            except Exception:
                LOG.exception('Final commit failed')
                session.rollback()

    return {'processed': processed, 'added_relations': added_relations}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description='Backfill author/genre master tables from Book rows')
    parser.add_argument('--dry-run', action='store_true', help='Do not write changes')
    parser.add_argument('--batch-size', type=int, default=200, help='Commit batch size')
    args = parser.parse_args(argv)

    LOG.info('Starting backfill: dry_run=%s batch_size=%d', args.dry_run, args.batch_size)
    res = backfill_all(batch_size=args.batch_size, dry_run=args.dry_run)
    LOG.info('Backfill finished: processed=%s added_relations=%s', res.get('processed'), res.get('added_relations'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
