"""
cli/display.py
==============
All terminal rendering for the Fuzzy RDB engine.

Uses the `rich` library for coloured tables and panels.
Falls back to plain ASCII if rich is not installed.

Public functions
----------------
  print_banner()
  print_results(df, title)         — ranked query results with membership bars
  print_tables(table_names)
  print_schema(table, columns)
  print_terms(table, registry)
  print_analysis(report)           — alpha-cut analysis report
  print_defuzz(col, term, values)  — defuzz method comparison
  print_suggestions(suggestions)
  print_ascii_plot(plot_str)
  print_help()
  print_error(msg)
  print_success(msg)
  print_info(msg)
"""

from __future__ import annotations
import sys
import math
from typing import Any

import pandas as pd

# ── Try to import rich; graceful fallback ─────────────────────────────────────
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    from rich.style import Style
    _RICH = True
except ImportError:
    _RICH = False


# ─────────────────────────────────────────────────────────────────────────────
# Console singleton
# ─────────────────────────────────────────────────────────────────────────────

if _RICH:
    console = Console()
else:
    class _FallbackConsole:
        def print(self, *args, **kwargs):
            # Strip rich markup from strings
            import re
            parts = []
            for a in args:
                s = str(a)
                s = re.sub(r'\[/?[^\]]*\]', '', s)
                parts.append(s)
            print(*parts)
        def rule(self, title="", **kwargs):
            width = 72
            if title:
                pad = (width - len(title) - 2) // 2
                print("─" * pad + " " + title + " " + "─" * pad)
            else:
                print("─" * width)
    console = _FallbackConsole()


# ─────────────────────────────────────────────────────────────────────────────
# Membership bar renderer
# ─────────────────────────────────────────────────────────────────────────────

def _membership_bar(mu: float, width: int = 20) -> str:
    """
    Render a horizontal bar representing a membership degree.

    0.0  ░░░░░░░░░░░░░░░░░░░░  0.00
    0.5  ██████████░░░░░░░░░░  0.50
    1.0  ████████████████████  1.00
    """
    mu = max(0.0, min(1.0, float(mu)))
    filled = round(mu * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"{bar}  {mu:.3f}"


def _color_for_mu(mu: float) -> str:
    """Return a rich color name based on membership degree."""
    if mu >= 0.7:
        return "green"
    elif mu >= 0.4:
        return "yellow"
    else:
        return "red"


# ─────────────────────────────────────────────────────────────────────────────
# Banner
# ─────────────────────────────────────────────────────────────────────────────

BANNER = """
 ╔═══════════════════════════════════════════════════╗
 ║       🌫  Fuzzy Relational Database Engine        ║
 ║    Extending SQL with Linguistic Intelligence     ║
 ╚═══════════════════════════════════════════════════╝

 Type  HELP  for a command reference.
 Type  EXIT  to quit.
"""

def print_banner():
    if _RICH:
        console.print(Panel(
            "[bold cyan]🌫  Fuzzy Relational Database Engine[/bold cyan]\n"
            "[dim]Extending SQL with Linguistic Intelligence[/dim]",
            border_style="cyan",
            expand=False,
        ))
        console.print()
        console.print("[dim]Type [bold]HELP[/bold] for commands · [bold]EXIT[/bold] to quit[/dim]\n")
    else:
        print(BANNER)


# ─────────────────────────────────────────────────────────────────────────────
# Query results
# ─────────────────────────────────────────────────────────────────────────────

def print_results(df: pd.DataFrame, title: str = "Query Results"):
    """Print a fuzzy query result DataFrame with membership bars."""
    if df is None or df.empty:
        print_info("No rows returned (try lowering THRESHOLD).")
        return

    data_cols = [c for c in df.columns if c != "_membership"]
    has_mu    = "_membership" in df.columns

    if _RICH:
        table = Table(
            title=f"[bold]{title}[/bold]",
            box=box.ROUNDED,
            show_lines=True,
            header_style="bold magenta",
        )

        for col in data_cols:
            table.add_column(col, style="white")
        if has_mu:
            table.add_column("Membership", style="cyan", min_width=28)

        for _, row in df.iterrows():
            cells = [str(row[c]) if not _is_float(row[c]) else f"{row[c]:.4g}"
                     for c in data_cols]
            if has_mu:
                mu  = float(row["_membership"])
                col = _color_for_mu(mu)
                bar = _membership_bar(mu)
                cells.append(f"[{col}]{bar}[/{col}]")
            table.add_row(*cells)

        console.print(table)
        if has_mu:
            console.print(
                f"[dim]  {len(df)} row(s) · "
                f"avg μ = {df['_membership'].mean():.3f} · "
                f"max μ = {df['_membership'].max():.3f}[/dim]\n"
            )
    else:
        # Plain ASCII fallback
        print(f"\n── {title} ──")
        headers = data_cols + (["Membership"] if has_mu else [])
        col_widths = [max(len(str(h)), 8) for h in headers]

        for col_i, col in enumerate(data_cols):
            col_widths[col_i] = max(col_widths[col_i],
                                    df[col].astype(str).str.len().max())
        if has_mu:
            col_widths[-1] = 28

        header_row = "  ".join(str(h).ljust(w) for h, w in zip(headers, col_widths))
        print(header_row)
        print("-" * len(header_row))

        for _, row in df.iterrows():
            vals = [str(row[c])[:col_widths[i]].ljust(col_widths[i])
                    for i, c in enumerate(data_cols)]
            if has_mu:
                vals.append(_membership_bar(float(row["_membership"])))
            print("  ".join(vals))
        print(f"\n{len(df)} row(s)\n")


# ─────────────────────────────────────────────────────────────────────────────
# Tables list
# ─────────────────────────────────────────────────────────────────────────────

def print_tables(table_names: list[str]):
    if not table_names:
        print_info("No tables loaded yet. Use  LOAD <path> AS <table>")
        return

    if _RICH:
        t = Table(title="Loaded Tables", box=box.SIMPLE_HEAD,
                  header_style="bold blue")
        t.add_column("#", style="dim", width=4)
        t.add_column("Table Name", style="bold cyan")
        for i, name in enumerate(table_names, 1):
            t.add_row(str(i), name)
        console.print(t)
    else:
        print("\n── Tables ──")
        for i, name in enumerate(table_names, 1):
            print(f"  {i}. {name}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Schema / DESCRIBE
# ─────────────────────────────────────────────────────────────────────────────

def print_schema(table: str, columns: dict[str, str],
                 stats: dict[str, dict] | None = None,
                 fuzzy_cols: list[str] | None = None):
    """
    Print table schema.
    columns   : {col_name: sqlite_type}
    stats     : optional {col_name: {min, max, mean, ...}} for numeric cols
    fuzzy_cols: columns that already have MFs defined
    """
    fuzzy_cols = fuzzy_cols or []

    if _RICH:
        t = Table(
            title=f"[bold]DESCRIBE  {table}[/bold]",
            box=box.ROUNDED,
            header_style="bold blue",
        )
        t.add_column("Column",  style="bold cyan")
        t.add_column("Type",    style="white")
        t.add_column("Fuzzy?",  style="green")
        if stats:
            t.add_column("Min",    style="dim")
            t.add_column("Max",    style="dim")
            t.add_column("Mean",   style="dim")

        for col, dtype in columns.items():
            is_fuzzy = "✓" if col in fuzzy_cols else ""
            row_cells = [col, dtype, is_fuzzy]
            if stats and col in stats:
                s = stats[col]
                row_cells += [
                    f"{s['min']:.4g}",
                    f"{s['max']:.4g}",
                    f"{s['mean']:.4g}",
                ]
            elif stats:
                row_cells += ["—", "—", "—"]
            t.add_row(*row_cells)
        console.print(t)
    else:
        print(f"\n── Schema: {table} ──")
        for col, dtype in columns.items():
            fz = " [fuzzy]" if col in fuzzy_cols else ""
            print(f"  {col:<20} {dtype:<10}{fz}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Membership function terms
# ─────────────────────────────────────────────────────────────────────────────

def print_terms(table: str, summary_rows: list[dict]):
    """
    Print all defined MF terms for a table.
    summary_rows: list of {column, term, type, params, universe}
    """
    if not summary_rows:
        print_info(
            f"No fuzzy terms defined for '{table}'. "
            "Use  DEFINE TERM <table>.<col> AS <term> USING <mf>(...)  to add one."
        )
        return

    if _RICH:
        t = Table(
            title=f"[bold]Fuzzy Terms: {table}[/bold]",
            box=box.ROUNDED,
            header_style="bold magenta",
        )
        t.add_column("Column",   style="cyan")
        t.add_column("Term",     style="bold green")
        t.add_column("MF Type",  style="white")
        t.add_column("Params",   style="dim")
        t.add_column("Universe", style="dim")
        for row in summary_rows:
            t.add_row(
                row["column"], row["term"],
                row["type"],   row["params"],
                row["universe"],
            )
        console.print(t)
    else:
        print(f"\n── Fuzzy Terms: {table} ──")
        for row in summary_rows:
            print(f"  {row['column']}.{row['term']}  "
                  f"[{row['type']}]  {row['params']}  U={row['universe']}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# ANALYZE report
# ─────────────────────────────────────────────────────────────────────────────

def print_analysis(report: dict):
    """Print the dict returned by alpha_cut.analyze()."""
    name = report.get("name", "A")

    if _RICH:
        t = Table(
            title=f"[bold]Fuzzy Set Analysis: {name}[/bold]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold blue",
        )
        t.add_column("Property",  style="cyan",  min_width=26)
        t.add_column("Value",     style="white", min_width=22)

        rows_to_show = [
            ("Height",              f"{report['height']:.4f}"),
            ("Normal?",             "✓ Yes" if report["is_normal"] else "✗ No"),
            ("Convex?",             "✓ Yes" if report["is_convex"] else "✗ No"),
            ("Support interval",    _fmt_interval(report["support_interval"])),
            ("Core interval",       _fmt_interval(report["core_interval"])),
            ("Cardinality (Σμ)",    f"{report['cardinality']:.4f}"),
            ("Relative cardinality",f"{report['relative_cardinality']:.4f}"),
            ("Bandwidth (α=0.5)",   f"{report['bandwidth_0.5']}" if report['bandwidth_0.5'] else "—"),
            ("Crossover pts (0.5)", str(report["crossover_pts_0.5"])),
            ("Entropy (De Luca-T)", f"{report['entropy_dlt']:.4f}"),
            ("Entropy (Yager)",     f"{report['entropy_yager']:.4f}"),
            ("Specificity",         f"{report['specificity']:.4f}"),
            ("Decomp. verified",    "✓ Yes" if report["decomp_verified"] else "✗ No"),
        ]
        for prop, val in rows_to_show:
            t.add_row(prop, val)
        console.print(t)
    else:
        print(f"\n── Analysis: {name} ──")
        for k, v in report.items():
            print(f"  {k:<28} {v}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Defuzz comparison
# ─────────────────────────────────────────────────────────────────────────────

def print_defuzz(col: str, term: str, values: dict[str, float]):
    """Print defuzz method comparison returned by compare_methods()."""
    if _RICH:
        t = Table(
            title=f"[bold]Defuzzification: {col} IS {term}[/bold]",
            box=box.ROUNDED,
            header_style="bold blue",
        )
        t.add_column("Method",       style="cyan")
        t.add_column("Crisp Output", style="bold yellow")
        for method, val in values.items():
            t.add_row(method, f"{val:.4f}")
        console.print(t)
    else:
        print(f"\n── Defuzz: {col} IS {term} ──")
        for method, val in values.items():
            print(f"  {method:<18} {val:.4f}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Suggestions
# ─────────────────────────────────────────────────────────────────────────────

def print_suggestions(suggestions: list[dict]):
    """Print data-driven MF suggestions."""
    if not suggestions:
        print_info("No suggestions generated.")
        return

    if _RICH:
        t = Table(
            title="[bold]Suggested MF Terms[/bold]",
            box=box.ROUNDED,
            header_style="bold magenta",
        )
        t.add_column("Column",  style="cyan")
        t.add_column("Term",    style="bold green")
        t.add_column("MF Type", style="white")
        t.add_column("Params",  style="dim")
        t.add_column("Copy this DEFINE command", style="yellow")
        for s in suggestions:
            params_str = ", ".join(f"{v:.2f}" for v in s["params"].values())
            cmd = (
                f"DEFINE TERM <table>.{s['column']} AS {s['term']} "
                f"USING {s['mf_type']}({params_str}) "
                f"UMIN {s['universe_min']:.2f} UMAX {s['universe_max']:.2f}"
            )
            t.add_row(
                s["column"], s["term"], s["mf_type"],
                str(s["params"]), cmd
            )
        console.print(t)
    else:
        print("\n── MF Suggestions ──")
        for s in suggestions:
            params_str = ", ".join(f"{v:.2f}" for v in s["params"].values())
            print(f"  {s['column']}.{s['term']}  [{s['mf_type']}]  {params_str}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# ASCII plot passthrough
# ─────────────────────────────────────────────────────────────────────────────

def print_ascii_plot(plot_str: str):
    if _RICH:
        console.print(Panel(plot_str, title="MF Plot", border_style="cyan"))
    else:
        print(plot_str)


# ─────────────────────────────────────────────────────────────────────────────
# HELP
# ─────────────────────────────────────────────────────────────────────────────

HELP_TEXT = """
[bold cyan]── FQL Command Reference ──────────────────────────────────────────[/bold cyan]

[bold yellow]Data loading[/bold yellow]
  LOAD <path.csv> AS <table>
  SHOW TABLES
  DESCRIBE <table>

[bold yellow]Define fuzzy terms[/bold yellow]
  DEFINE TERM <table>.<col> AS <term> USING <mf_type>(<params>)
         [UMIN <n> UMAX <n>]

  MF types & params:
    triangular(a, b, c)            trapezoidal(a, b, c, d)
    gaussian(mean, sigma)          generalized_bell(a, b, c)
    sigmoid(c, a)                  s_shaped(a, b)
    z_shaped(a, b)                 pi_shaped(a, b, c, d)
    singleton(center, tolerance)

  Hedges (use in WHERE clause):
    very · extremely · somewhat · quite · slightly
    indeed · plus · minus · more_or_less · not · not_very

  SHOW TERMS <table>
  SUGGEST TERMS <table>.<col>  [N <int>]
  PLOT <table>.<col>

[bold yellow]Query[/bold yellow]
  SELECT * FROM <table>
    WHERE <col> IS [<hedge>] <term>
    [AND | OR <col> IS [<hedge>] <term>] ...
    [THRESHOLD <0.0 – 1.0>]
    [TOP <n>]

[bold yellow]Analysis[/bold yellow]
  ANALYZE <table>.<col> IS <term>
  DEFUZZ  <table>.<col> IS <term> METHOD <centroid|bisector|mom|som|lom>

[bold yellow]Other[/bold yellow]
  HELP    EXIT

[dim]── Tips ────────────────────────────────────────────────────────────
  • THRESHOLD 0.0 returns all rows ranked by degree.
  • THRESHOLD 0.5 returns only rows with μ ≥ 0.5.
  • Use TOP 10 to cap the number of results shown.
  • Combine hedges: WHERE age IS very young AND salary IS not low
──────────────────────────────────────────────────────────────────[/dim]
"""

HELP_TEXT_PLAIN = """
── FQL Command Reference ──────────────────────────────────────────

Data loading
  LOAD <path.csv> AS <table>
  SHOW TABLES
  DESCRIBE <table>

Define fuzzy terms
  DEFINE TERM <table>.<col> AS <term> USING <mf_type>(<params>)
         [UMIN <n> UMAX <n>]

  MF types:  triangular | trapezoidal | gaussian | generalized_bell
             sigmoid | s_shaped | z_shaped | pi_shaped | singleton

  Hedges:  very · extremely · somewhat · quite · slightly
           not · more_or_less

  SHOW TERMS <table>
  SUGGEST TERMS <table>.<col> [N <int>]
  PLOT <table>.<col>

Query
  SELECT * FROM <table>
    WHERE <col> IS [<hedge>] <term>
    [AND | OR <col> IS [<hedge>] <term>] ...
    [THRESHOLD <0.0-1.0>]
    [TOP <n>]

Analysis
  ANALYZE <table>.<col> IS <term>
  DEFUZZ  <table>.<col> IS <term> METHOD <centroid|bisector|mom|som|lom>

Other
  HELP    EXIT
"""

def print_help():
    if _RICH:
        console.print(HELP_TEXT)
    else:
        print(HELP_TEXT_PLAIN)


# ─────────────────────────────────────────────────────────────────────────────
# Status messages
# ─────────────────────────────────────────────────────────────────────────────

def print_error(msg: str):
    if _RICH:
        console.print(f"[bold red]✖  Error:[/bold red]  {msg}")
    else:
        print(f"ERROR: {msg}", file=sys.stderr)


def print_success(msg: str):
    if _RICH:
        console.print(f"[bold green]✔  {msg}[/bold green]")
    else:
        print(f"OK: {msg}")


def print_info(msg: str):
    if _RICH:
        console.print(f"[dim]ℹ  {msg}[/dim]")
    else:
        print(f"  {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_float(val: Any) -> bool:
    try:
        float(val)
        return True
    except (TypeError, ValueError):
        return False


def _fmt_interval(interval) -> str:
    if interval is None:
        return "—"
    lo, hi = interval
    return f"[{lo:.4g}, {hi:.4g}]"