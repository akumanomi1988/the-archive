# filepath: d:\Proyectos\skald\scripts\inspect_sqlite_db.py
# Small diagnostic tool: lists tables, row counts, PRAGMA journal_mode, integrity_check
# Usage:
#   .venv\Scripts\Activate.ps1
#   python scripts\inspect_sqlite_db.py path\to\skald.db
#   python scripts\inspect_sqlite_db.py path\to\skald.db --checkpoint
from pathlib import Path
import sqlite3
import sys

def _human_size(n: int) -> str:
	if n < 1024:
		return f"{n} B"
	for unit in ("KB", "MB", "GB", "TB"):
		n /= 1024.0
		if n < 1024.0:
			return f"{n:.2f} {unit}"
	return f"{n:.2f} PB"

def inspect(db_path: Path, do_checkpoint: bool = False) -> int:
	if not db_path.exists():
		print("File not found:", db_path)
		return 1
	print("DB file:", db_path.resolve())
	print("Size bytes:", db_path.stat().st_size, "(", _human_size(db_path.stat().st_size), ")")
	conn = sqlite3.connect(str(db_path))
	try:
		cur = conn.cursor()
		# PRAGMAs
		try:
			cur.execute("PRAGMA journal_mode;")
			jm = cur.fetchone()
			print("journal_mode:", jm[0] if jm else "<unknown>")
		except Exception as e:
			print("journal_mode failed:", e)
		try:
			cur.execute("PRAGMA page_count;")
			pc = cur.fetchone()
			print("page_count:", pc[0] if pc else "<unknown>")
		except Exception as e:
			print("page_count failed:", e)
		try:
			cur.execute("PRAGMA page_size;")
			ps = cur.fetchone()
			print("page_size:", ps[0] if ps else "<unknown>")
		except Exception as e:
			print("page_size failed:", e)
		# integrity
		try:
			cur.execute("PRAGMA integrity_check;")
			res = cur.fetchone()[0]
			print("integrity_check:", res)
		except Exception as e:
			print("integrity_check failed:", e)
		# Tables / Views
		try:
			cur.execute("""SELECT name, type FROM sqlite_master WHERE type IN ('table','view') ORDER BY name;""")
			rows = cur.fetchall()
		except Exception as e:
			print("Failed to read sqlite_master:", e)
			rows = []
		if not rows:
			print("No tables or views found in sqlite_master.")
		else:
			print("\nTables / Views:")
			for name, typ in rows:
				print(" -", name, f"({typ})")
			print("\nRow counts (may be slow):")
			for name, typ in rows:
				# skip sqlite internal tables that cannot be counted directly
				if name.startswith("sqlite_"):
					print(f"  {name}: skipped (internal)")
					continue
				try:
					# Use double quotes to allow unusual names
					cur.execute(f'SELECT COUNT(*) FROM "{name}";')
					c = cur.fetchone()[0]
				except Exception as e:
					c = f"error: {e}"
				print(f"  {name}: {c}")
		# WAL files (use name-based suffixing to support files without dot-suffix)
		wal = db_path.with_name(db_path.name + "-wal")
		shm = db_path.with_name(db_path.name + "-shm")
		print("\nWAL present:", wal.exists(), "SHM present:", shm.exists())
		if do_checkpoint:
			print("\nPerforming WAL checkpoint (TRUNCATE)...")
			try:
				cur.execute("PRAGMA wal_checkpoint(TRUNCATE);")
				# fetch results if any
				try:
					res = cur.fetchall()
					if res:
						print("Checkpoint result:", res)
					else:
						print("Checkpoint executed.")
				except Exception:
					print("Checkpoint executed.")
				conn.commit()
			except Exception as e:
				print("Checkpoint failed:", e)
	finally:
		conn.close()
	return 0

if __name__ == '__main__':
	import argparse
	p = argparse.ArgumentParser(description="Inspect a sqlite DB file (tables, row counts, PRAGMA, WAL).")
	p.add_argument('db', help='Path to sqlite db')
	p.add_argument('--checkpoint', action='store_true', help='Run PRAGMA wal_checkpoint(TRUNCATE)')
	args = p.parse_args()
	sys.exit(inspect(Path(args.db), args.checkpoint))