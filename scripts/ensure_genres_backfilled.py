"""Ensure Genre master table and BookGenre relations are populated.

This script is idempotent: it will print counts, run the backfill logic (calling
`backend.migrate_backfill`) if there are no genres or no relations, and then print
post-run counts.

Run from repo root with the project venv active.
"""
import sys
from pathlib import Path
from sqlmodel import Session, select, func

# Ensure repo root on sys.path for direct script execution
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import backend

def main():
    backend.init_app()
    with Session(backend.ENGINE) as s:
        def _scalar_count(stmt):
            r = s.exec(stmt).one()
            # r may be an int or a tuple like (count,)
            return int(r if isinstance(r, int) else (r[0] if isinstance(r, (list, tuple)) else r))

        total_books = _scalar_count(select(func.count()).select_from(backend.Book))
        total_genres = _scalar_count(select(func.count()).select_from(backend.Genre))
        total_bookgenre = _scalar_count(select(func.count()).select_from(backend.BookGenre))
        books_with_genre = _scalar_count(select(func.count()).select_from(backend.Book).where(backend.Book.genre is not None))
        print(f"Books={total_books}, Genres={total_genres}, BookGenre={total_bookgenre}, Books.with_genre={books_with_genre}")
        # If genres or relations seem missing, run backfill
        if total_genres == 0 or total_bookgenre == 0:
            print("Running migrate_backfill to populate Genre and BookGenre tables...")
            res = backend.migrate_backfill(session=s)
            print("Backfill result:", res)
            # Recompute
            total_genres = int(s.exec(select(func.count()).select_from(backend.Genre)).scalar_one())
            total_bookgenre = int(s.exec(select(func.count()).select_from(backend.BookGenre)).scalar_one())
            print(f"After backfill -> Genres={total_genres}, BookGenre={total_bookgenre}")
        else:
            print("No backfill needed.")

if __name__ == '__main__':
    main()
