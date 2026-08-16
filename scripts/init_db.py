"""Create the local SQLite schema used by the DataStore."""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "data_layer" / "sqlite_schema.sql"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db",
        default=str(ROOT / "database" / "aic2026.sqlite"),
        help="SQLite database path",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA.read_text(encoding="utf-8"))
    print(f"Initialized SQLite database: {db_path}")


if __name__ == "__main__":
    main()
