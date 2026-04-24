"""
storage.py — SQLite Storage Layer for Fuzzy RDB Engine
========================================================
Responsibilities:
  - Create / drop tables dynamically from user schema
  - Load CSV files into tables
  - Insert, fetch, list records
  - Store membership function definitions as metadata
  - Keep everything in a single .db file (or in-memory for testing)
"""

import sqlite3
import csv
import json
import os
import pandas as pd
from typing import Any


# ─────────────────────────────────────────────
#  Storage Engine
# ─────────────────────────────────────────────

class StorageEngine:
    """
    Manages all SQLite interactions for the Fuzzy RDB Engine.

    One StorageEngine = one .db file.
    All user tables live inside it alongside a hidden _mf_registry
    table that stores membership function definitions.
    """

    def __init__(self, db_path: str = ":memory:"):
        """
        Parameters
        ----------
        db_path : str
            Path to the SQLite file. Use ":memory:" for a temporary
            in-memory database (good for testing / fresh demos).
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row          # rows behave like dicts
        self._init_metadata_tables()

    # ── Internal Setup ──────────────────────────────────────────────────────

    def _init_metadata_tables(self):
        """Create engine-internal metadata tables if they don't exist yet."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS _mf_registry (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                table_name  TEXT    NOT NULL,
                column_name TEXT    NOT NULL,
                term        TEXT    NOT NULL,        -- e.g. 'young', 'high'
                mf_type     TEXT    NOT NULL,        -- triangular | trapezoidal | gaussian | sigmoid
                params      TEXT    NOT NULL,        -- JSON array of MF parameters
                UNIQUE(table_name, column_name, term)
            );

            CREATE TABLE IF NOT EXISTS _table_meta (
                table_name  TEXT PRIMARY KEY,
                column_info TEXT NOT NULL,           -- JSON: {col: dtype, ...}
                created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """)
        self.conn.commit()

    # ── Table Management ────────────────────────────────────────────────────

    def create_table(self, table_name: str, columns: dict[str, str]) -> None:
        """
        Dynamically create a table.

        Parameters
        ----------
        table_name : str
            Name for the new table.
        columns : dict
            {column_name: sqlite_type}  e.g. {"age": "REAL", "name": "TEXT"}
        """
        _validate_identifier(table_name)
        col_defs = ", ".join(
            f"{_quote(col)} {dtype}" for col, dtype in columns.items()
        )
        sql = f"CREATE TABLE IF NOT EXISTS {_quote(table_name)} ({col_defs});"
        self.conn.execute(sql)

        # Store column metadata
        self.conn.execute(
            "INSERT OR REPLACE INTO _table_meta (table_name, column_info) VALUES (?, ?)",
            (table_name, json.dumps(columns))
        )
        self.conn.commit()
        print(f"[Storage] Table '{table_name}' created with columns: {list(columns.keys())}")

    def drop_table(self, table_name: str) -> None:
        """Drop a user table and its associated metadata."""
        _validate_identifier(table_name)
        self.conn.execute(f"DROP TABLE IF EXISTS {_quote(table_name)};")
        self.conn.execute("DELETE FROM _table_meta WHERE table_name = ?", (table_name,))
        self.conn.execute("DELETE FROM _mf_registry WHERE table_name = ?", (table_name,))
        self.conn.commit()
        print(f"[Storage] Table '{table_name}' dropped.")

    def list_tables(self) -> list[str]:
        """Return all user-created table names (excludes internal _ tables)."""
        cur = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE '\\_%' ESCAPE '\\';"
        )
        return [row[0] for row in cur.fetchall()]

    def get_columns(self, table_name: str) -> dict[str, str]:
        """
        Return column names and their SQLite types for a table.
        Returns {col_name: col_type}
        """
        _validate_identifier(table_name)
        cur = self.conn.execute(f"PRAGMA table_info({_quote(table_name)});")
        return {row["name"]: row["type"] for row in cur.fetchall()}

    def get_numeric_columns(self, table_name: str) -> list[str]:
        """Return only REAL / INTEGER columns — candidates for fuzzification."""
        cols = self.get_columns(table_name)
        return [c for c, t in cols.items() if t.upper() in ("REAL", "INTEGER", "NUMERIC", "FLOAT")]

    # ── Data Insertion ───────────────────────────────────────────────────────

    def insert_row(self, table_name: str, row: dict) -> None:
        """Insert a single row dict into a table."""
        _validate_identifier(table_name)
        cols = ", ".join(_quote(k) for k in row.keys())
        placeholders = ", ".join("?" for _ in row)
        sql = f"INSERT INTO {_quote(table_name)} ({cols}) VALUES ({placeholders});"
        self.conn.execute(sql, list(row.values()))
        self.conn.commit()

    def insert_many(self, table_name: str, rows: list[dict]) -> None:
        """Bulk insert a list of row dicts."""
        if not rows:
            return
        _validate_identifier(table_name)
        cols = ", ".join(_quote(k) for k in rows[0].keys())
        placeholders = ", ".join("?" for _ in rows[0])
        sql = f"INSERT INTO {_quote(table_name)} ({cols}) VALUES ({placeholders});"
        self.conn.executemany(sql, [list(r.values()) for r in rows])
        self.conn.commit()
        print(f"[Storage] Inserted {len(rows)} rows into '{table_name}'.")

    # ── CSV Loading ──────────────────────────────────────────────────────────

    def load_csv(
        self,
        filepath: str,
        table_name: str,
        infer_types: bool = True
    ) -> pd.DataFrame:
        """
        Load a CSV file into a SQLite table.

        - Auto-infers column types (numeric → REAL, else TEXT).
        - Creates the table automatically.
        - Returns the loaded data as a DataFrame for preview.

        Parameters
        ----------
        filepath    : path to the CSV file
        table_name  : desired table name
        infer_types : if True, numeric columns become REAL; else all TEXT
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"CSV not found: {filepath}")

        df = pd.read_csv(filepath)
        df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

        # Infer SQLite types
        columns = {}
        for col in df.columns:
            if infer_types and pd.api.types.is_numeric_dtype(df[col]):
                columns[col] = "REAL"
            else:
                columns[col] = "TEXT"

        self.create_table(table_name, columns)

        # Insert rows
        rows = df.where(pd.notnull(df), None).to_dict(orient="records")
        self.insert_many(table_name, rows)

        print(f"[Storage] Loaded '{filepath}' → table '{table_name}' ({len(df)} rows, {len(df.columns)} cols)")
        return df

    def load_dataframe(self, df: pd.DataFrame, table_name: str) -> None:
        """Load a pandas DataFrame directly into a table."""
        df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
        columns = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                columns[col] = "REAL"
            else:
                columns[col] = "TEXT"
        self.create_table(table_name, columns)
        rows = df.where(pd.notnull(df), None).to_dict(orient="records")
        self.insert_many(table_name, rows)

    # ── Data Retrieval ───────────────────────────────────────────────────────

    def fetch_all(self, table_name: str) -> list[dict]:
        """Return all rows from a table as a list of dicts."""
        _validate_identifier(table_name)
        cur = self.conn.execute(f"SELECT * FROM {_quote(table_name)};")
        return [dict(row) for row in cur.fetchall()]

    def fetch_as_dataframe(self, table_name: str) -> pd.DataFrame:
        """Return all rows as a pandas DataFrame."""
        return pd.read_sql_query(
            f"SELECT * FROM {_quote(table_name)};",
            self.conn
        )

    def execute_sql(self, sql: str, params: tuple = ()) -> list[dict]:
        """
        Run an arbitrary SELECT statement.
        Use this for the classical SQL comparison module.
        """
        cur = self.conn.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]

    def row_count(self, table_name: str) -> int:
        _validate_identifier(table_name)
        cur = self.conn.execute(f"SELECT COUNT(*) FROM {_quote(table_name)};")
        return cur.fetchone()[0]

    # ── Membership Function Metadata ─────────────────────────────────────────

    def save_mf(
        self,
        table_name: str,
        column_name: str,
        term: str,
        mf_type: str,
        params: list[float]
    ) -> None:
        """
        Persist a membership function definition to the registry.

        Example:
            save_mf("employees", "age", "young", "triangular", [0, 25, 35])
        """
        self.conn.execute("""
            INSERT OR REPLACE INTO _mf_registry
                (table_name, column_name, term, mf_type, params)
            VALUES (?, ?, ?, ?, ?)
        """, (table_name, column_name, term, mf_type, json.dumps(params)))
        self.conn.commit()

    def get_mfs_for_column(self, table_name: str, column_name: str) -> list[dict]:
        """Return all MF definitions for a specific column."""
        cur = self.conn.execute("""
            SELECT term, mf_type, params
            FROM _mf_registry
            WHERE table_name = ? AND column_name = ?
        """, (table_name, column_name))
        results = []
        for row in cur.fetchall():
            results.append({
                "term": row["term"],
                "mf_type": row["mf_type"],
                "params": json.loads(row["params"])
            })
        return results

    def get_all_mfs(self, table_name: str) -> dict[str, list[dict]]:
        """
        Return all MF definitions for a table, grouped by column.
        Returns: {column_name: [{term, mf_type, params}, ...]}
        """
        cur = self.conn.execute("""
            SELECT column_name, term, mf_type, params
            FROM _mf_registry
            WHERE table_name = ?
            ORDER BY column_name, term
        """, (table_name,))
        result: dict[str, list] = {}
        for row in cur.fetchall():
            col = row["column_name"]
            result.setdefault(col, []).append({
                "term": row["term"],
                "mf_type": row["mf_type"],
                "params": json.loads(row["params"])
            })
        return result

    def delete_mf(self, table_name: str, column_name: str, term: str) -> None:
        """Remove a single MF definition."""
        self.conn.execute("""
            DELETE FROM _mf_registry
            WHERE table_name=? AND column_name=? AND term=?
        """, (table_name, column_name, term))
        self.conn.commit()

    # ── Utilities ────────────────────────────────────────────────────────────

    def get_column_range(self, table_name: str, column_name: str) -> tuple[float, float]:
        """Return (min, max) of a numeric column — useful for MF design."""
        _validate_identifier(table_name)
        _validate_identifier(column_name)
        cur = self.conn.execute(
            f"SELECT MIN({_quote(column_name)}), MAX({_quote(column_name)}) "
            f"FROM {_quote(table_name)};"
        )
        row = cur.fetchone()
        return (float(row[0]), float(row[1]))

    def column_stats(self, table_name: str, column_name: str) -> dict:
        """Return basic stats for a numeric column."""
        df = self.fetch_as_dataframe(table_name)
        col = df[column_name]
        return {
            "min":    float(col.min()),
            "max":    float(col.max()),
            "mean":   float(col.mean()),
            "median": float(col.median()),
            "std":    float(col.std()),
            "count":  int(col.count()),
        }

    def close(self):
        """Close the database connection."""
        self.conn.close()
        print("[Storage] Connection closed.")

    def __repr__(self):
        tables = self.list_tables()
        return f"<StorageEngine db='{self.db_path}' tables={tables}>"


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def _quote(identifier: str) -> str:
    """Wrap a SQL identifier in double-quotes to handle reserved words/spaces."""
    return f'"{identifier}"'


def _validate_identifier(name: str) -> None:
    """Basic guard against SQL injection in table/column names."""
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
    if not all(c in allowed for c in name):
        raise ValueError(
            f"Invalid identifier '{name}'. Use only letters, digits, and underscores."
        )
