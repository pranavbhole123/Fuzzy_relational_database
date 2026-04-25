"""
cli/parser.py
=============
Fuzzy Query Language (FQL) parser.

Converts raw input strings into structured command dicts that
shell.py dispatches to the FuzzyRDBEngine.

──────────────────────────────────────────────────────────────
SUPPORTED SYNTAX  (keywords are case-insensitive)
──────────────────────────────────────────────────────────────

  LOAD  <csv_path>  AS  <table>

  SHOW  TABLES
  SHOW  TERMS  <table>

  DESCRIBE  <table>

  DEFINE TERM <table>.<col> AS <term>
         USING <mf_type>(<p1>, <p2>, ...)
         [UMIN <number>  UMAX <number>]

  SELECT  *
          FROM  <table>
          [WHERE <col> IS [<hedge>] <term>
                 [AND | OR  <col> IS [<hedge>] <term>] ...]
          [THRESHOLD  <float>]
          [TOP  <int>]

  ANALYZE  <table>.<col>  IS  <term>

  DEFUZZ  <table>.<col>  IS  <term>
          METHOD  <centroid | bisector | mom | som | lom>

  SUGGEST TERMS <table>.<col>  [N <int>]

  PLOT  <table>.<col>

  HELP
  EXIT | QUIT
──────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import re
import shlex
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# MF positional-param → named-param mapping
# Must match membership_functions.py MF_PARAM_GUIDE
# ─────────────────────────────────────────────────────────────────────────────

MF_PARAM_KEYS: dict[str, list[str]] = {
    "triangular"       : ["a", "b", "c"],
    "trapezoidal"      : ["a", "b", "c", "d"],
    "gaussian"         : ["mean", "sigma"],
    "generalized_bell" : ["a", "b", "c"],
    "sigmoid"          : ["c", "a"],
    "s_shaped"         : ["a", "b"],
    "z_shaped"         : ["a", "b"],
    "pi_shaped"        : ["a", "b", "c", "d"],
    "singleton"        : ["center", "tolerance"],
}

KNOWN_HEDGES = {
    "very", "extremely", "indeed", "plus",
    "somewhat", "quite", "slightly", "minus",
    "more_or_less", "not", "not_very",
}

KNOWN_MF_TYPES = set(MF_PARAM_KEYS.keys())

KNOWN_DEFUZZ_METHODS = {"centroid", "bisector", "mom", "som", "lom", "weighted_avg"}


# ─────────────────────────────────────────────────────────────────────────────
# Exception
# ─────────────────────────────────────────────────────────────────────────────

class ParseError(Exception):
    """Raised for any FQL syntax error."""


# ─────────────────────────────────────────────────────────────────────────────
# Token stream helper
# ─────────────────────────────────────────────────────────────────────────────

class _Tokens:
    def __init__(self, tokens: list[str]):
        self._t = tokens
        self._i = 0

    def peek(self, offset: int = 0) -> str | None:
        idx = self._i + offset
        return self._t[idx].upper() if idx < len(self._t) else None

    def peek_raw(self, offset: int = 0) -> str | None:
        idx = self._i + offset
        return self._t[idx] if idx < len(self._t) else None

    def consume(self) -> str:
        if self._i >= len(self._t):
            raise ParseError("Unexpected end of input.")
        v = self._t[self._i]
        self._i += 1
        return v

    def consume_upper(self) -> str:
        return self.consume().upper()

    def consume_raw(self) -> str:
        return self.consume()

    def expect(self, *keywords: str) -> str:
        v = self.consume()
        if v.upper() not in [k.upper() for k in keywords]:
            raise ParseError(
                f"Expected {' or '.join(repr(k) for k in keywords)}, got {v!r}."
            )
        return v

    def has(self) -> bool:
        return self._i < len(self._t)

    def remaining_raw(self) -> list[str]:
        return self._t[self._i:]


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse(line: str) -> dict[str, Any]:
    """
    Parse a single FQL line.  Returns a dict with a 'cmd' key.
    Raises ParseError for unrecognised or malformed input.
    """
    line = line.strip()
    if not line or line.startswith("--"):
        return {"cmd": "noop"}

    try:
        raw_tokens = shlex.split(line)
    except ValueError:
        raw_tokens = line.split()

    if not raw_tokens:
        return {"cmd": "noop"}

    tok = _Tokens(raw_tokens)
    kw = tok.peek()

    if kw == "LOAD":
        return _parse_load(tok)
    elif kw == "SHOW":
        return _parse_show(tok)
    elif kw == "DESCRIBE":
        return _parse_describe(tok)
    elif kw == "DEFINE":
        return _parse_define(tok)
    elif kw == "SELECT":
        return _parse_select(tok)
    elif kw == "ANALYZE":
        return _parse_analyze(tok)
    elif kw == "DEFUZZ":
        return _parse_defuzz(tok)
    elif kw == "SUGGEST":
        return _parse_suggest(tok)
    elif kw == "PLOT":
        return _parse_plot(tok)
    elif kw in ("HELP", "?"):
        return {"cmd": "help"}
    elif kw in ("EXIT", "QUIT", "BYE"):
        return {"cmd": "exit"}
    else:
        raise ParseError(
            f"Unknown command: {tok.peek_raw()!r}. "
            "Type HELP to see available commands."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Individual command parsers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_load(tok: _Tokens) -> dict:
    tok.expect("LOAD")
    path = tok.consume_raw()
    tok.expect("AS")
    table = tok.consume_raw()
    return {"cmd": "load", "path": path, "table": table}


def _parse_show(tok: _Tokens) -> dict:
    tok.expect("SHOW")
    what = tok.peek()
    if what == "TABLES":
        tok.consume()
        return {"cmd": "show_tables"}
    elif what == "TERMS":
        tok.consume()
        if not tok.has():
            raise ParseError("SHOW TERMS requires a table name.")
        table = tok.consume_raw()
        return {"cmd": "show_terms", "table": table}
    else:
        raise ParseError(
            f"SHOW requires TABLES or TERMS <table>, got {tok.peek_raw()!r}."
        )


def _parse_describe(tok: _Tokens) -> dict:
    tok.expect("DESCRIBE")
    if not tok.has():
        raise ParseError("DESCRIBE requires a table name.")
    table = tok.consume_raw()
    return {"cmd": "describe", "table": table}


def _parse_define(tok: _Tokens) -> dict:
    """
    DEFINE TERM <table>.<col> AS <term>
           USING <mf_type>(<p1>, <p2>, ...)
           [UMIN <n> UMAX <n>]
    """
    tok.expect("DEFINE")
    tok.expect("TERM")

    ref = tok.consume_raw()
    if "." not in ref:
        raise ParseError(
            f"Expected <table>.<column>, got {ref!r}. "
            "Example:  DEFINE TERM employees.age AS young USING triangular(15, 25, 35)"
        )
    table, col = ref.split(".", 1)

    tok.expect("AS")
    term = tok.consume_raw()
    tok.expect("USING")

    # Rejoin remaining tokens — shlex may have split the MF spec
    remainder = " ".join(tok.remaining_raw())
    mf_type, params_dict, umin, umax = _parse_mf_spec(remainder)

    return {
        "cmd"    : "define",
        "table"  : table,
        "col"    : col,
        "term"   : term,
        "mf_type": mf_type,
        "params" : params_dict,
        "umin"   : umin,   # None → infer from data column range
        "umax"   : umax,
    }


def _parse_mf_spec(text: str) -> tuple:
    """
    Parse:  <mf_type>(<p1>, <p2>, ...) [UMIN <n> UMAX <n>]

    Returns (mf_type_str, params_dict, umin_or_None, umax_or_None)
    """
    umin = umax = None

    umin_m = re.search(r'\bUMIN\s+([\d.eE+\-]+)', text, re.IGNORECASE)
    umax_m = re.search(r'\bUMAX\s+([\d.eE+\-]+)', text, re.IGNORECASE)

    if umin_m:
        umin = float(umin_m.group(1))
        text = text[:umin_m.start()].strip()
    if umax_m:
        umax = float(umax_m.group(1))
        text = re.sub(r'\bUMAX\s+[\d.eE+\-]+', '', text, flags=re.IGNORECASE).strip()

    m = re.match(r'([a-zA-Z_]+)\s*\(([^)]*)\)', text.strip())
    if not m:
        raise ParseError(
            f"Cannot parse MF spec: {text!r}\n"
            "Expected:  mf_type(p1, p2, ...)  "
            "e.g.  triangular(0, 25, 35)"
        )

    mf_type = m.group(1).lower()
    raw_params = [p.strip() for p in m.group(2).split(",") if p.strip()]

    if mf_type not in KNOWN_MF_TYPES:
        raise ParseError(
            f"Unknown MF type: {mf_type!r}. "
            f"Valid: {sorted(KNOWN_MF_TYPES)}"
        )

    keys = MF_PARAM_KEYS[mf_type]
    if len(raw_params) != len(keys):
        raise ParseError(
            f"{mf_type} requires {len(keys)} param(s) ({', '.join(keys)}), "
            f"got {len(raw_params)}."
        )

    try:
        params_dict = {k: float(v) for k, v in zip(keys, raw_params)}
    except ValueError as exc:
        raise ParseError(f"Non-numeric MF parameter: {exc}")

    return mf_type, params_dict, umin, umax


def _parse_select(tok: _Tokens) -> dict:
    """
    SELECT * FROM <table>
    [WHERE <col> IS [<hedge>] <term> [AND|OR ...] ...]
    [THRESHOLD <float>]
    [TOP <int>]
    """
    tok.expect("SELECT")
    tok.consume_raw()  # * or column list — we accept it, engine uses all cols
    tok.expect("FROM")
    if not tok.has():
        raise ParseError("FROM requires a table name.")
    table = tok.consume_raw()

    conditions = []
    threshold  = 0.0
    top_k      = None

    while tok.has():
        kw = tok.peek()
        if kw == "WHERE":
            tok.consume()
            conditions = _parse_where_conditions(tok)
        elif kw == "THRESHOLD":
            tok.consume()
            try:
                threshold = float(tok.consume_raw())
            except ValueError:
                raise ParseError("THRESHOLD requires a float (0.0 – 1.0).")
        elif kw == "TOP":
            tok.consume()
            try:
                top_k = int(tok.consume_raw())
            except ValueError:
                raise ParseError("TOP requires an integer.")
        else:
            raise ParseError(
                f"Unexpected token in SELECT: {tok.peek_raw()!r}. "
                "Expected WHERE, THRESHOLD, or TOP."
            )

    return {
        "cmd"       : "select",
        "table"     : table,
        "conditions": conditions,
        "threshold" : threshold,
        "top_k"     : top_k,
    }


def _parse_where_conditions(tok: _Tokens) -> list[dict]:
    """
    Parse chained fuzzy conditions:
      <col> IS [<hedge>] <term> [AND|OR <col> IS [<hedge>] <term>] ...
    """
    conditions = []
    logic = "AND"

    while tok.has():
        if tok.peek() in ("THRESHOLD", "TOP"):
            break

        col = tok.consume_raw()
        tok.expect("IS")

        hedge = None
        if tok.has() and tok.peek() and tok.peek().lower() in KNOWN_HEDGES:
            hedge = tok.consume_raw().lower()

        if not tok.has():
            raise ParseError(
                f"Missing term name after 'IS' for column {col!r}."
            )
        term = tok.consume_raw()

        conditions.append({
            "col"  : col,
            "hedge": hedge,
            "term" : term,
            "logic": logic,
        })

        if tok.has() and tok.peek() in ("AND", "OR"):
            logic = tok.consume_upper()
        else:
            break

    if not conditions:
        raise ParseError("WHERE clause has no conditions.")
    return conditions


def _parse_analyze(tok: _Tokens) -> dict:
    """ANALYZE <table>.<col> IS <term>"""
    tok.expect("ANALYZE")
    ref = tok.consume_raw()
    if "." not in ref:
        raise ParseError(f"Expected <table>.<column>, got {ref!r}.")
    table, col = ref.split(".", 1)
    tok.expect("IS")
    term = tok.consume_raw()
    return {"cmd": "analyze", "table": table, "col": col, "term": term}


def _parse_defuzz(tok: _Tokens) -> dict:
    """DEFUZZ <table>.<col> IS <term> METHOD <method>"""
    tok.expect("DEFUZZ")
    ref = tok.consume_raw()
    if "." not in ref:
        raise ParseError(f"Expected <table>.<column>, got {ref!r}.")
    table, col = ref.split(".", 1)
    tok.expect("IS")
    term = tok.consume_raw()
    tok.expect("METHOD")
    method = tok.consume_raw().lower()
    if method not in KNOWN_DEFUZZ_METHODS:
        raise ParseError(
            f"Unknown defuzz method {method!r}. "
            f"Valid: {sorted(KNOWN_DEFUZZ_METHODS)}"
        )
    return {
        "cmd"   : "defuzz",
        "table" : table,
        "col"   : col,
        "term"  : term,
        "method": method,
    }


def _parse_suggest(tok: _Tokens) -> dict:
    """SUGGEST TERMS <table>.<col> [N <int>]"""
    tok.expect("SUGGEST")
    tok.expect("TERMS")
    ref = tok.consume_raw()
    if "." not in ref:
        raise ParseError(f"Expected <table>.<column>, got {ref!r}.")
    table, col = ref.split(".", 1)
    n = 3
    if tok.has() and tok.peek() == "N":
        tok.consume()
        try:
            n = int(tok.consume_raw())
        except ValueError:
            raise ParseError("N requires an integer.")
    return {"cmd": "suggest", "table": table, "col": col, "n": n}


def _parse_plot(tok: _Tokens) -> dict:
    """PLOT <table>.<col>"""
    tok.expect("PLOT")
    ref = tok.consume_raw()
    if "." not in ref:
        raise ParseError(f"Expected <table>.<column>, got {ref!r}.")
    table, col = ref.split(".", 1)
    return {"cmd": "plot", "table": table, "col": col}