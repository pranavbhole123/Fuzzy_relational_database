"""
cli/shell.py
============
Interactive REPL for the Fuzzy Relational Database Engine.

Run from the project root:
    python cli/shell.py
    python cli/shell.py --db fuzzy.db          # persistent database
    python cli/shell.py --script demo.fql       # run a .fql script file

Query lifecycle:
    user types FQL
        ↓  parser.py  tokenises → command dict
        ↓  shell      dispatches to FuzzyRDBEngine method
        ↓  engine     resolves MFs / hedges via fuzzifier.py
        ↓  fuzzy_sql  executes WHERE / JOIN / GROUP BY / SET OPS
        ↓  display.py renders rich tables + membership bars
"""

from __future__ import annotations
import os
import sys
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, os.path.join(_ROOT, "core")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cli.parser  import parse, ParseError, MF_PARAM_KEYS
from cli.display import (
    print_banner, print_results, print_tables,
    print_schema, print_terms, print_analysis,
    print_defuzz, print_suggestions, print_ascii_plot,
    print_help, print_error, print_success, print_info,
    console,
)
from main import FuzzyRDBEngine


class FuzzyShell:
    """Interactive REPL for the Fuzzy RDB Engine."""

    PROMPT = "\n[fql]> "

    def __init__(self, db_path: str = ":memory:"):
        self.engine   = FuzzyRDBEngine(db_path=db_path)
        self.history  : list[str] = []
        self._running = True

    # ─────────────────────────────────────────────────────────────────────────
    # Run modes
    # ─────────────────────────────────────────────────────────────────────────

    def run_interactive(self):
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
        try:
            cmd = parse(line)
        except ParseError as exc:
            print_error(str(exc))
            return

        handler = getattr(self, f"_cmd_{cmd['cmd']}", None)
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
            print_error(f"{type(exc).__name__}: {exc}")

    # ─────────────────────────────────────────────────────────────────────────
    # Utility helpers (used by multiple handlers)
    # ─────────────────────────────────────────────────────────────────────────

    def _membership_summary(self, result) -> str:
        """One-line stats about the membership column of a result."""
        if result.empty or "_membership" not in result.columns:
            return ""
        m = result["_membership"]
        return (
            f"{len(result)} rows  |  "
            f"μ  min={m.min():.3f}  mean={m.mean():.3f}  max={m.max():.3f}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Universal commands
    # ─────────────────────────────────────────────────────────────────────────

    def _cmd_noop(self, _cmd):
        pass

    def _cmd_exit(self, _cmd=None):
        print_info("Goodbye.")
        self._running = False

    def _cmd_help(self, _cmd=None):
        print_help()

    # ─────────────────────────────────────────────────────────────────────────
    # SHOW variants
    # ─────────────────────────────────────────────────────────────────────────

    def _cmd_show_tables(self, _cmd: dict):
        print_tables(self.engine.list_tables())

    def _cmd_show_terms(self, cmd: dict):
        table   = cmd["table"]
        summary = self.engine.list_terms(table)
        print_terms(table, summary)

    def _cmd_show_hedges(self, _cmd: dict):
        """SHOW HEDGES — list all linguistic modifiers."""
        hedges = self.engine.list_hedges()
        print_info("Available linguistic hedges:")
        print()
        # reuse print_results with a manufactured DataFrame
        import pandas as pd
        df = pd.DataFrame(hedges)
        print_results(df, title="Linguistic Hedges")
        print_info(
            "Usage:  SELECT * FROM <table> WHERE <col> IS very <term>\n"
            "        SELECT * FROM <table> WHERE <col> IS not <term>"
        )

    def _cmd_show_mf_types(self, _cmd: dict):
        """SHOW MF_TYPES — list every supported membership function."""
        mf_types = self.engine.list_mf_types()
        import pandas as pd
        df = pd.DataFrame(mf_types)
        print_results(df, title="Supported Membership Function Types")
        print_info(
            "Usage:  DEFINE TERM <table>.<col> AS <term> USING <mf_type>(<params>)\n"
            "Example: DEFINE TERM employees.age AS young USING triangular(15, 25, 35)"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Data loading
    # ─────────────────────────────────────────────────────────────────────────

    def _cmd_load(self, cmd: dict):
        path  = cmd["path"]
        table = cmd["table"]

        if not os.path.exists(path):
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
        desc = self.engine.describe_table(table)
        print_schema(table, desc["columns"],
                     stats=desc["stats"], fuzzy_cols=desc["fuzzy_cols"])

    # ─────────────────────────────────────────────────────────────────────────
    # Table introspection
    # ─────────────────────────────────────────────────────────────────────────

    def _cmd_describe(self, cmd: dict):
        table = cmd["table"]
        desc  = self.engine.describe_table(table)
        print_schema(table, desc["columns"],
                     stats=desc["stats"], fuzzy_cols=desc["fuzzy_cols"])
        print_info(f"{desc['row_count']} rows in table '{table}'.")

    # ─────────────────────────────────────────────────────────────────────────
    # MF management
    # ─────────────────────────────────────────────────────────────────────────

    def _cmd_define(self, cmd: dict):
        self.engine.define_term(
            table_name=cmd["table"], column=cmd["col"],
            term=cmd["term"],        mf_type=cmd["mf_type"],
            params=cmd["params"],    umin=cmd.get("umin"),
            umax=cmd.get("umax"),
        )
        params_str = ", ".join(f"{k}={v}" for k, v in cmd["params"].items())
        print_success(
            f"Defined  {cmd['table']}.{cmd['col']} IS '{cmd['term']}'  "
            f"[{cmd['mf_type']}({params_str})]"
        )

    def _cmd_suggest(self, cmd: dict):
        suggestions = self.engine.suggest_terms(
            cmd["table"], cmd["col"], n_terms=cmd.get("n", 3)
        )
        print_suggestions(suggestions)
        print_info(
            "To register a suggestion copy and run a DEFINE TERM command.\n"
            "Adjust the parameters to match your domain knowledge."
        )

    def _cmd_plot(self, cmd: dict):
        plot_str = self.engine.ascii_plot(cmd["table"], cmd["col"])
        print_ascii_plot(plot_str)

    # ─────────────────────────────────────────────────────────────────────────
    # SELECT  (handles basic, BETWEEN, IN, DISTINCT variants)
    # ─────────────────────────────────────────────────────────────────────────

    def _cmd_select(self, cmd: dict):
        table     = cmd["table"]
        threshold = cmd.get("threshold", 0.0)
        top_k     = cmd.get("top_k")

        # ── BETWEEN variant ──────────────────────────────────────────────────
        if cmd.get("between"):
            b      = cmd["between"]
            result = self.engine.query_between(
                table, b["col"], b["low"], b["high"],
                softness=b.get("softness", 0.1),
                threshold=threshold, top_k=top_k,
            )
            title = (
                f"SELECT * FROM {table} WHERE "
                f"{b['col']} BETWEEN {b['low']} AND {b['high']}  "
                f"[softness={b.get('softness', 0.1)}]"
            )
            print_results(result, title=title)
            print_info(self._membership_summary(result))
            return

        # ── IN variant ───────────────────────────────────────────────────────
        if cmd.get("fuzzy_in"):
            fi     = cmd["fuzzy_in"]
            result = self.engine.query_in(
                table, fi["col"], fi["values"],
                tolerance=fi.get("tolerance", 0.0),
                threshold=threshold, top_k=top_k,
            )
            vals_str = ", ".join(str(v) for v in fi["values"])
            title = (
                f"SELECT * FROM {table} WHERE "
                f"{fi['col']} IN ({vals_str})  "
                f"[tolerance={fi.get('tolerance', 0.0)}]"
            )
            print_results(result, title=title)
            print_info(self._membership_summary(result))
            return

        # ── DISTINCT variant ─────────────────────────────────────────────────
        if cmd.get("distinct"):
            result = self.engine.query_distinct(
                table,
                columns=cmd.get("distinct_on") or None,
                similarity=cmd.get("similarity", 0.95),
                conditions=cmd.get("conditions") or None,
                threshold=threshold,
            )
            title = (
                f"SELECT DISTINCT * FROM {table}  "
                f"[similarity≤{cmd.get('similarity', 0.95)}]"
            )
            print_results(result, title=title)
            print_info(
                f"Near-duplicates removed. "
                f"{self._membership_summary(result)}"
            )
            return

        # ── Plain SELECT * (no WHERE) ────────────────────────────────────────
        conditions = cmd.get("conditions", [])
        if not conditions:
            df = self.engine.storage.fetch_as_dataframe(table)
            print_results(df, title=f"SELECT * FROM {table}")
            print_info(f"{len(df)} rows.")
            return

        # ── Standard fuzzy WHERE ─────────────────────────────────────────────
        result = self.engine.query(
            table_name=table, conditions=conditions,
            threshold=threshold, top_k=top_k,
        )

        where_parts = []
        for c in conditions:
            hedge = f" {c['hedge']}" if c.get("hedge") else ""
            where_parts.append(f"{c['col']} IS{hedge} {c['term']}")
        sep   = " AND " if conditions[0].get("logic", "AND") == "AND" else " OR "
        title = (
            f"SELECT * FROM {table} WHERE " + sep.join(where_parts)
            + (f"  THRESHOLD {threshold}" if threshold > 0 else "")
        )

        print_results(result, title=title)
        print_info(self._membership_summary(result))

        if result.empty:
            print_info(
                "No rows above threshold. "
                "Try lowering THRESHOLD or use SUGGEST TERMS to verify your MF."
            )

    # ─────────────────────────────────────────────────────────────────────────
    # JOIN
    # ─────────────────────────────────────────────────────────────────────────

    def _cmd_join(self, cmd: dict):
        """
        JOIN <left> WITH <right> ON <col>
             [TYPE inner|left|right|full]
             [TOLERANCE <float>]  [THRESHOLD <float>]  [TOP <int>]
        """
        lt        = cmd["left_table"]
        rt        = cmd["right_table"]
        on        = cmd["on"]
        join_type = cmd.get("join_type", "inner")
        tolerance = cmd.get("tolerance", 0.0)
        threshold = cmd.get("threshold", 0.0)
        top_k     = cmd.get("top_k")

        result = self.engine.join(
            left_table=lt, right_table=rt,
            on=on, join_type=join_type,
            tolerance=tolerance, threshold=threshold, top_k=top_k,
        )

        tol_str  = f"  tolerance={tolerance}" if tolerance > 0 else ""
        title    = (
            f"JOIN {lt} ⋈ {rt}  ON {on}  "
            f"[{join_type.upper()}{tol_str}]"
        )
        print_results(result, title=title)
        print_info(self._membership_summary(result))

        # Viva hint: tolerance > 0 means fuzzy matching was used
        if tolerance > 0:
            classical = sum(1 for _ in result.iterrows()
                            if result["_membership"].iloc[0] == 1.0)
            print_info(
                f"Fuzzy join matched {len(result)} pairs. "
                f"Classical JOIN (tolerance=0) would have returned only "
                f"exact key matches."
            )

    # ─────────────────────────────────────────────────────────────────────────
    # GROUP BY
    # ─────────────────────────────────────────────────────────────────────────

    def _cmd_groupby(self, cmd: dict):
        """
        GROUPBY <table> ON <col> TERMS <t1>,<t2>,...
                [AGGREGATE <agg_col> FUNCS <f1>,<f2>,...]
        """
        table    = cmd["table"]
        col      = cmd["col"]
        terms    = cmd["terms"]
        agg_col  = cmd.get("agg_col")
        agg_funcs= cmd.get("agg_funcs", ["fuzzy_count", "weighted_avg", "avg"])
        min_mem  = cmd.get("min_mem", 0.0)

        result = self.engine.group_by(
            table_name=table, col=col, terms=terms,
            agg_col=agg_col, agg_funcs=agg_funcs,
            min_membership=min_mem,
        )

        groups      = result["groups"]
        aggregation = result["aggregation"]

        print_info(
            f"Fuzzy GROUP BY  {table}.{col}  "
            f"→  {len(terms)} groups  "
            f"(rows can belong to multiple groups)"
        )

        # Print each group's top rows
        for term, gdf in groups.items():
            print_results(
                gdf.head(5),
                title=f"Group: {col} IS '{term}'  ({len(gdf)} rows ≥ {min_mem})"
            )

        # Print aggregation if computed
        if aggregation is not None and not aggregation.empty:
            print_results(
                aggregation,
                title=(
                    f"Aggregation on '{agg_col}'  "
                    f"[{', '.join(agg_funcs)}]"
                )
            )
            print_info(
                "fuzzy_count = Σμ(x)  (weighted count by membership degree)\n"
                "weighted_avg = Σ(μ·v) / Σμ  (membership-weighted average)"
            )

    # ─────────────────────────────────────────────────────────────────────────
    # SET OPERATIONS
    # ─────────────────────────────────────────────────────────────────────────

    def _cmd_set_op(self, cmd: dict):
        """
        UNION / INTERSECT / EXCEPT
        Operates on loaded tables or previously stored query results.
        """
        op       = cmd["op"]          # union | intersect | except
        table_a  = cmd["table_a"]
        table_b  = cmd["table_b"]
        key      = cmd["key"]
        threshold= cmd.get("threshold", 0.0)

        result = self.engine.set_op(
            op=op, table_a=table_a, table_b=table_b,
            key=key, threshold=threshold,
        )

        op_symbols = {"union": "∪", "intersect": "∩", "except": "∖"}
        title = (
            f"{table_a}  {op_symbols.get(op, op.upper())}  {table_b}  "
            f"ON {key}  [{op.upper()}]"
        )
        print_results(result, title=title)
        print_info(self._membership_summary(result))

        # Explain what the operation means
        explanations = {
            "union"    : "UNION: max(μA, μB) per row — broadest coverage",
            "intersect": "INTERSECT: min(μA, μB) per row — only well-matched rows from both",
            "except"   : "EXCEPT: rows in A not well-matched in B — μ = μA × (1 − best_match_B)",
        }
        print_info(explanations.get(op, ""))

    # ─────────────────────────────────────────────────────────────────────────
    # COMPARE  (the viva proof command)
    # ─────────────────────────────────────────────────────────────────────────

    def _cmd_compare(self, cmd: dict):
        """
        COMPARE <table>
                FUZZY   <conditions>
                AGAINST <classical conditions>
                [THRESHOLD <float>]

        The centrepiece of the viva demonstration.
        Shows exactly what fuzzy found that classical SQL missed.
        """
        comparison = self.engine.compare(
            table_name           = cmd["table"],
            fuzzy_conditions     = cmd["fuzzy_conditions"],
            classical_conditions = cmd["classical_conditions"],
            threshold            = cmd.get("threshold", 0.0),
        )

        # ── Classical result ─────────────────────────────────────────────────
        print_results(
            comparison["classical_result"],
            title=(
                f"CLASSICAL SQL result  "
                f"({comparison['classical_count']} rows)  "
                f"— binary, no ranking"
            )
        )

        # ── Fuzzy result ─────────────────────────────────────────────────────
        print_results(
            comparison["fuzzy_result"],
            title=(
                f"FUZZY result  "
                f"({comparison['fuzzy_count']} rows)  "
                f"— ranked by membership degree"
            )
        )

        # ── Rows only fuzzy found ────────────────────────────────────────────
        only_fuzzy = comparison["only_in_fuzzy"]
        if not only_fuzzy.empty:
            print_results(
                only_fuzzy,
                title=(
                    f"Rows ONLY fuzzy found  "
                    f"({len(only_fuzzy)})  "
                    f"— classical SQL missed these entirely"
                )
            )
        else:
            print_info("Classical SQL captured the same rows as fuzzy (no exclusive fuzzy finds).")

        # ── Rows only classical found ────────────────────────────────────────
        only_classical = comparison["only_in_classical"]
        if not only_classical.empty:
            print_results(
                only_classical,
                title=(
                    f"Rows ONLY classical found  ({len(only_classical)})  "
                    f"— fuzzy scored these below threshold"
                )
            )

        # ── Summary panel ────────────────────────────────────────────────────
        print_info("=" * 60)
        print_info("COMPARISON SUMMARY")
        print_info("=" * 60)
        print_info(f"  Classical SQL  : {comparison['classical_count']} rows  (binary yes/no)")
        print_info(f"  Fuzzy query    : {comparison['fuzzy_count']} rows  (ranked by degree)")
        print_info(f"  Only in fuzzy  : {len(only_fuzzy)}  (missed by classical)")
        print_info(f"  Only in classical: {len(only_classical)}  (below fuzzy threshold)")
        print_info("")
        print_info(comparison["fuzzy_advantage"])
        print_info(
            "\nKey insight: Classical SQL uses hard cutoffs — a row either qualifies\n"
            "or not. Fuzzy SQL ranks borderline cases instead of discarding them,\n"
            "revealing relevant results that classical queries miss entirely."
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Analysis commands
    # ─────────────────────────────────────────────────────────────────────────

    def _cmd_analyze(self, cmd: dict):
        report = self.engine.analyze(cmd["table"], cmd["col"], cmd["term"])
        print_analysis(report)

    def _cmd_defuzz(self, cmd: dict):
        method = cmd.get("method", "all")
        if method == "all":
            values = self.engine.defuzz(
                cmd["table"], cmd["col"], cmd["term"], method="all"
            )
        else:
            crisp  = self.engine.defuzz(
                cmd["table"], cmd["col"], cmd["term"], method=method
            )
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
    p.add_argument("--db", default=":memory:",
                   help="SQLite database path (default: in-memory, ephemeral)")
    p.add_argument("--script", default=None,
                   help="Path to a .fql script file to run then exit")
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