"""
cli/shell.py
============
Interactive REPL for the Fuzzy Relational Database Engine.

Run from the project root:
    python cli/shell.py
    python cli/shell.py --db fuzzy.db          # persistent database
    python cli/shell.py --script demo.fql       # run a .fql script file

The shell ties together:
    cli/parser.py   → parse FQL input into command dicts
    main.py         → FuzzyRDBEngine that executes every command
    cli/display.py  → rich terminal rendering of all outputs

Query lifecycle (matching the architecture diagram):
    user types FQL
        ↓
    parser.py tokenises → command dict
        ↓
    shell dispatches to FuzzyRDBEngine method
        ↓
    engine resolves terms → MF objects + hedges (via fuzzifier.py)
        ↓
    fuzzy_sql.py executes WHERE / JOIN / GROUP BY
        ↓
    alpha_cut.py on ANALYZE command
        ↓
    display.py renders rich tables + membership bars
"""

from __future__ import annotations
import os
import sys
import argparse

# ── Path setup so imports work from any working directory ────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, os.path.join(_ROOT, "core")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cli.parser  import parse, ParseError          # noqa: E402
from cli.display import (                          # noqa: E402
    print_banner, print_results, print_tables,
    print_schema, print_terms, print_analysis,
    print_defuzz, print_suggestions, print_ascii_plot,
    print_help, print_error, print_success, print_info,
    console,
)
from main import FuzzyRDBEngine                    # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Shell
# ─────────────────────────────────────────────────────────────────────────────

class FuzzyShell:
    """
    Interactive REPL for the Fuzzy RDB Engine.

    Attributes
    ----------
    engine  : FuzzyRDBEngine
    prompt  : str
    history : list of input lines (for this session)
    """

    PROMPT = "\n[fql]> "

    def __init__(self, db_path: str = ":memory:"):
        self.engine  = FuzzyRDBEngine(db_path=db_path)
        self.history : list[str] = []
        self._running = True

    # ─────────────────────────────────────────────────────────────────────────
    # Run modes
    # ─────────────────────────────────────────────────────────────────────────

    def run_interactive(self):
        """Start the REPL."""
        print_banner()
        while self._running:
            try:
                line = input(self.PROMPT).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                self._cmd_exit()
                break

            if not line:
                continue

            self.history.append(line)
            self._execute_line(line)

    def run_script(self, path: str):
        """Execute a .fql script file non-interactively."""
        if not os.path.exists(path):
            print_error(f"Script file not found: {path!r}")
            return

        print_info(f"Running script: {path}")
        with open(path) as fh:
            for lineno, raw_line in enumerate(fh, 1):
                line = raw_line.strip()
                if not line or line.startswith("--"):
                    continue
                print_info(f"  [{lineno}] {line}")
                self._execute_line(line)

    # ─────────────────────────────────────────────────────────────────────────
    # Dispatch
    # ─────────────────────────────────────────────────────────────────────────

    def _execute_line(self, line: str):
        """Parse one line and dispatch to the appropriate handler."""
        try:
            cmd = parse(line)
        except ParseError as exc:
            print_error(str(exc))
            return

        handler_name = f"_cmd_{cmd['cmd']}"
        handler = getattr(self, handler_name, None)

        if handler is None:
            print_error(f"No handler for command '{cmd['cmd']}'.")
            return

        try:
            handler(cmd)
        except KeyError as exc:
            print_error(str(exc))
        except ValueError as exc:
            print_error(str(exc))
        except FileNotFoundError as exc:
            print_error(str(exc))
        except Exception as exc:
            print_error(f"Unexpected error: {type(exc).__name__}: {exc}")

    # ─────────────────────────────────────────────────────────────────────────
    # Command handlers
    # ─────────────────────────────────────────────────────────────────────────

    def _cmd_noop(self, _cmd):
        pass

    def _cmd_exit(self, _cmd=None):
        print_info("Goodbye.")
        self._running = False

    def _cmd_help(self, _cmd=None):
        print_help()

    # ── Data ─────────────────────────────────────────────────────────────────

    def _cmd_load(self, cmd: dict):
        path  = cmd["path"]
        table = cmd["table"]

        if not os.path.exists(path):
            # Try relative to project root
            alt = os.path.join(_ROOT, path)
            if os.path.exists(alt):
                path = alt
            else:
                print_error(f"File not found: {path!r}")
                return

        df = self.engine.load_csv(path, table)
        print_success(
            f"Loaded '{os.path.basename(path)}' → table '{table}'  "
            f"({len(df)} rows, {len(df.columns)} columns)"
        )
        # Show a quick preview
        desc = self.engine.describe_table(table)
        print_schema(
            table,
            desc["columns"],
            stats     = desc["stats"],
            fuzzy_cols= desc["fuzzy_cols"],
        )

    def _cmd_show_tables(self, _cmd: dict):
        print_tables(self.engine.list_tables())

    def _cmd_describe(self, cmd: dict):
        table = cmd["table"]
        desc  = self.engine.describe_table(table)
        print_schema(
            table,
            desc["columns"],
            stats     = desc["stats"],
            fuzzy_cols= desc["fuzzy_cols"],
        )
        print_info(f"{desc['row_count']} rows in table '{table}'.")

    # ── Fuzzy term management ─────────────────────────────────────────────────

    def _cmd_define(self, cmd: dict):
        self.engine.define_term(
            table_name = cmd["table"],
            column     = cmd["col"],
            term       = cmd["term"],
            mf_type    = cmd["mf_type"],
            params     = cmd["params"],
            umin       = cmd.get("umin"),
            umax       = cmd.get("umax"),
        )
        params_str = ", ".join(f"{k}={v}" for k, v in cmd["params"].items())
        print_success(
            f"Defined:  {cmd['table']}.{cmd['col']} IS {cmd['term']}  "
            f"[{cmd['mf_type']}]  {{{params_str}}}"
        )

    def _cmd_show_terms(self, cmd: dict):
        table   = cmd["table"]
        summary = self.engine.list_terms(table)
        print_terms(table, summary)

    def _cmd_suggest(self, cmd: dict):
        suggestions = self.engine.suggest_terms(
            cmd["table"], cmd["col"], n_terms=cmd.get("n", 3)
        )
        print_suggestions(suggestions)
        print_info(
            "Copy a DEFINE TERM command above to register a suggestion, "
            "then adjust params as needed."
        )

    def _cmd_plot(self, cmd: dict):
        plot_str = self.engine.ascii_plot(cmd["table"], cmd["col"])
        print_ascii_plot(plot_str)

    # ── Query ─────────────────────────────────────────────────────────────────

    def _cmd_select(self, cmd: dict):
        table      = cmd["table"]
        conditions = cmd["conditions"]
        threshold  = cmd.get("threshold", 0.0)
        top_k      = cmd.get("top_k")

        if not conditions:
            # Plain SELECT * FROM table — show all rows, no membership
            df = self.engine.storage.fetch_as_dataframe(table)
            print_results(df, title=f"SELECT * FROM {table}")
            return

        result = self.engine.query(
            table_name = table,
            conditions = conditions,
            threshold  = threshold,
            top_k      = top_k,
        )

        # Build a human-readable title
        where_parts = []
        for c in conditions:
            hedge = f" {c['hedge']}" if c.get("hedge") else ""
            where_parts.append(f"{c['col']} IS{hedge} {c['term']}")
        connector = " AND " if conditions[0]["logic"] == "AND" else " OR "
        title = (
            f"SELECT * FROM {table} WHERE "
            + connector.join(where_parts)
            + (f"  THRESHOLD {threshold}" if threshold > 0 else "")
        )

        print_results(result, title=title)

        if result.empty:
            print_info(
                "No results above threshold. "
                "Try  THRESHOLD 0.0  to see all rows ranked."
            )

    # ── Analysis ──────────────────────────────────────────────────────────────

    def _cmd_analyze(self, cmd: dict):
        report = self.engine.analyze(cmd["table"], cmd["col"], cmd["term"])
        print_analysis(report)

    def _cmd_defuzz(self, cmd: dict):
        method = cmd.get("method", "all")
        if method == "all":
            values = self.engine.defuzz(cmd["table"], cmd["col"], cmd["term"], method="all")
        else:
            crisp  = self.engine.defuzz(cmd["table"], cmd["col"], cmd["term"], method=method)
            values = {method: crisp}
        print_defuzz(cmd["col"], cmd["term"], values)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_cli_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fuzzy Relational Database Engine — interactive shell",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli/shell.py
  python cli/shell.py --db mydata.db
  python cli/shell.py --script demo.fql
  python cli/shell.py --db mydata.db --script setup.fql
        """,
    )
    p.add_argument(
        "--db", default=":memory:",
        help="SQLite database path (default: :memory:, ephemeral)",
    )
    p.add_argument(
        "--script", default=None,
        help="Path to a .fql script file to run then exit",
    )
    return p.parse_args()


def main():
    args  = _parse_cli_args()
    shell = FuzzyShell(db_path=args.db)

    if args.script:
        shell.run_script(args.script)
    else:
        shell.run_interactive()


if __name__ == "__main__":
    main()