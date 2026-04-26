"""
cli/parser.py
=============
Fuzzy Query Language (FQL) parser.

Converts raw input strings into structured command dicts that
shell.py dispatches to the FuzzyRDBEngine.

──────────────────────────────────────────────────────────────
FULL SUPPORTED SYNTAX  (keywords are case-insensitive)
──────────────────────────────────────────────────────────────

DATA LOADING
  LOAD  <csv_path>  AS  <table>

INTROSPECTION
  SHOW  TABLES
  SHOW  TERMS     <table>
  SHOW  HEDGES
  SHOW  MF_TYPES
  DESCRIBE  <table>

MF DEFINITION
  DEFINE TERM <table>.<col> AS <term>
         USING <mf_type>(<p1>, <p2>, ...)
         [UMIN <number>  UMAX <number>]

  SUGGEST TERMS <table>.<col>  [N <int>]

─────────────────────────────────────────────────────────────
SELECT  (basic fuzzy query)
  SELECT  *  FROM  <table>
    [WHERE <col> IS [<hedge>] <term>
           [AND|OR  <col> IS [<hedge>] <term>] ...]
    [THRESHOLD  <float>]
    [TOP  <int>]

SELECT BETWEEN  (soft range query)
  SELECT  *  FROM  <table>
    WHERE <col> BETWEEN <low> AND <high>
    [SOFTNESS <float>]
    [THRESHOLD <float>]
    [TOP <int>]

SELECT DISTINCT  (remove near-duplicates)
  SELECT DISTINCT *  FROM  <table>
    [ON <col1>, <col2>, ...]
    [SIMILARITY <float>]
    [WHERE  ...]
    [THRESHOLD <float>]

SELECT IN  (fuzzy set membership)
  SELECT  *  FROM  <table>
    WHERE <col> IN (<v1>, <v2>, <v3>)
    [TOLERANCE <float>]
    [THRESHOLD <float>]

─────────────────────────────────────────────────────────────
JOIN
  JOIN  <left_table>  WITH  <right_table>
        ON  <col>
        [TYPE  inner | left | right | full]
        [TOLERANCE  <float>]
        [THRESHOLD  <float>]
        [TOP  <int>]

─────────────────────────────────────────────────────────────
GROUP BY + AGGREGATE
  GROUPBY  <table>  ON  <col>
           TERMS  <term1>, <term2>, ...
           [AGGREGATE  <agg_col>
            FUNCS  <func1>, <func2>, ...]

─────────────────────────────────────────────────────────────
SET OPERATIONS
  UNION      <table1>  AND  <table2>  ON  <key>
  INTERSECT  <table1>  AND  <table2>  ON  <key>
  EXCEPT     <table1>  FROM <table2>  ON  <key>

─────────────────────────────────────────────────────────────
ANALYSIS
  ANALYZE  <table>.<col>  IS  <term>
  DEFUZZ   <table>.<col>  IS  <term>
           METHOD  <centroid | bisector | mom | som | lom>
  PLOT     <table>.<col>

─────────────────────────────────────────────────────────────
VIVA PROOF
  COMPARE  <table>
           FUZZY   <col> IS [<hedge>] <term> [AND|OR ...] ...
           AGAINST <col> <op> <value>       [AND      ...] ...
           [THRESHOLD <float>]

─────────────────────────────────────────────────────────────
  HELP
  EXIT | QUIT
──────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import re
import shlex
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Constants
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

KNOWN_JOIN_TYPES = {"inner", "left", "right", "full"}

KNOWN_AGG_FUNCS = {
    "count", "fuzzy_count", "sum", "weighted_sum",
    "avg", "weighted_avg", "min", "max", "std",
}

CLASSICAL_OPS = {"<", "<=", ">", ">=", "==", "!=", "="}

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

    def consume_float(self, label: str) -> float:
        try:
            return float(self.consume_raw())
        except ValueError:
            raise ParseError(f"{label} requires a float value.")

    def consume_int(self, label: str) -> int:
        try:
            return int(self.consume_raw())
        except ValueError:
            raise ParseError(f"{label} requires an integer value.")


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
    kw  = tok.peek()

    dispatch = {
        "LOAD"     : _parse_load,
        "SHOW"     : _parse_show,
        "DESCRIBE" : _parse_describe,
        "DEFINE"   : _parse_define,
        "SELECT"   : _parse_select,
        "JOIN"     : _parse_join,
        "GROUPBY"  : _parse_groupby,
        "GROUP"    : _parse_groupby,        # alias: GROUP BY → GROUPBY
        "UNION"    : _parse_set_op,
        "INTERSECT": _parse_set_op,
        "EXCEPT"   : _parse_set_op,
        "ANALYZE"  : _parse_analyze,
        "DEFUZZ"   : _parse_defuzz,
        "SUGGEST"  : _parse_suggest,
        "PLOT"     : _parse_plot,
        "COMPARE"  : _parse_compare,
        "HELP"     : lambda t: {"cmd": "help"},
        "?"        : lambda t: {"cmd": "help"},
        "EXIT"     : lambda t: {"cmd": "exit"},
        "QUIT"     : lambda t: {"cmd": "exit"},
        "BYE"      : lambda t: {"cmd": "exit"},
    }

    if kw not in dispatch:
        raise ParseError(
            f"Unknown command: {tok.peek_raw()!r}. "
            "Type HELP to see available commands."
        )

    return dispatch[kw](tok)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────────────────

def _parse_load(tok: _Tokens) -> dict:
    tok.expect("LOAD")
    path = tok.consume_raw()
    tok.expect("AS")
    table = tok.consume_raw()
    return {"cmd": "load", "path": path, "table": table}


# ─────────────────────────────────────────────────────────────────────────────
# SHOW
# ─────────────────────────────────────────────────────────────────────────────

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

    elif what == "HEDGES":
        tok.consume()
        return {"cmd": "show_hedges"}

    elif what in ("MF_TYPES", "MFTYPES", "MF"):
        tok.consume()
        return {"cmd": "show_mf_types"}

    else:
        raise ParseError(
            f"SHOW requires TABLES | TERMS <table> | HEDGES | MF_TYPES, "
            f"got {tok.peek_raw()!r}."
        )


# ─────────────────────────────────────────────────────────────────────────────
# DESCRIBE
# ─────────────────────────────────────────────────────────────────────────────

def _parse_describe(tok: _Tokens) -> dict:
    tok.expect("DESCRIBE")
    if not tok.has():
        raise ParseError("DESCRIBE requires a table name.")
    table = tok.consume_raw()
    return {"cmd": "describe", "table": table}


# ─────────────────────────────────────────────────────────────────────────────
# DEFINE TERM
# ─────────────────────────────────────────────────────────────────────────────

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
            f"Expected <table>.<column>, got {ref!r}.\n"
            "Example:  DEFINE TERM employees.age AS young USING triangular(15, 25, 35)"
        )
    table, col = ref.split(".", 1)

    tok.expect("AS")
    term = tok.consume_raw()
    tok.expect("USING")

    remainder = " ".join(tok.remaining_raw())
    mf_type, params_dict, umin, umax = _parse_mf_spec(remainder)

    return {
        "cmd"    : "define",
        "table"  : table,
        "col"    : col,
        "term"   : term,
        "mf_type": mf_type,
        "params" : params_dict,
        "umin"   : umin,
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
            "Expected:  mf_type(p1, p2, ...)  e.g.  triangular(0, 25, 35)"
        )

    mf_type    = m.group(1).lower()
    raw_params = [p.strip() for p in m.group(2).split(",") if p.strip()]

    if mf_type not in KNOWN_MF_TYPES:
        raise ParseError(
            f"Unknown MF type: {mf_type!r}.\n"
            f"Valid types: {sorted(KNOWN_MF_TYPES)}"
        )

    keys = MF_PARAM_KEYS[mf_type]
    if len(raw_params) != len(keys):
        raise ParseError(
            f"{mf_type} needs {len(keys)} param(s) ({', '.join(keys)}), "
            f"got {len(raw_params)}."
        )

    try:
        params_dict = {k: float(v) for k, v in zip(keys, raw_params)}
    except ValueError as exc:
        raise ParseError(f"Non-numeric MF parameter: {exc}")

    return mf_type, params_dict, umin, umax


# ─────────────────────────────────────────────────────────────────────────────
# SELECT  (handles basic, BETWEEN, DISTINCT, IN variants)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_select(tok: _Tokens) -> dict:
    tok.expect("SELECT")

    # Check for DISTINCT modifier
    distinct     = False
    distinct_on  = []
    similarity   = 0.95

    if tok.peek() == "DISTINCT":
        tok.consume()
        distinct = True

    tok.consume_raw()   # * or column list — engine uses all cols
    tok.expect("FROM")
    if not tok.has():
        raise ParseError("FROM requires a table name.")
    table = tok.consume_raw()

    conditions  = []
    threshold   = 0.0
    top_k       = None
    between     = None      # dict if BETWEEN query
    fuzzy_in    = None      # dict if IN query

    while tok.has():
        kw = tok.peek()

        if kw == "WHERE":
            tok.consume()
            # Peek ahead: BETWEEN or IN or standard IS conditions?
            if not tok.has():
                raise ParseError("WHERE clause is empty.")

            # Check for BETWEEN:  WHERE col BETWEEN low AND high
            if tok.peek(1) == "BETWEEN":
                col    = tok.consume_raw()
                tok.expect("BETWEEN")
                low    = tok.consume_float("BETWEEN lower bound")
                tok.expect("AND")
                high   = tok.consume_float("BETWEEN upper bound")
                soft   = 0.1
                if tok.has() and tok.peek() == "SOFTNESS":
                    tok.consume()
                    soft = tok.consume_float("SOFTNESS")
                between = {"col": col, "low": low, "high": high, "softness": soft}

            # Check for IN:  WHERE col IN (v1, v2, v3)
            elif tok.peek(1) == "IN":
                col = tok.consume_raw()
                tok.expect("IN")
                raw_list = " ".join(tok.remaining_raw())
                values, tok = _parse_in_list(raw_list, tok)
                tolerance = 0.0
                if tok.has() and tok.peek() == "TOLERANCE":
                    tok.consume()
                    tolerance = tok.consume_float("TOLERANCE")
                fuzzy_in = {"col": col, "values": values, "tolerance": tolerance}

            else:
                conditions = _parse_where_conditions(tok)

        elif kw == "THRESHOLD":
            tok.consume()
            threshold = tok.consume_float("THRESHOLD")

        elif kw == "TOP":
            tok.consume()
            top_k = tok.consume_int("TOP")

        elif kw == "ON" and distinct:
            tok.consume()
            raw_cols = " ".join(tok.remaining_raw()).split(",")
            distinct_on = [c.strip() for c in raw_cols if c.strip()]
            break

        elif kw == "SIMILARITY" and distinct:
            tok.consume()
            similarity = tok.consume_float("SIMILARITY")

        else:
            raise ParseError(
                f"Unexpected token in SELECT: {tok.peek_raw()!r}. "
                "Expected WHERE, THRESHOLD, TOP, ON, or SIMILARITY."
            )

    return {
        "cmd"        : "select",
        "table"      : table,
        "conditions" : conditions,
        "threshold"  : threshold,
        "top_k"      : top_k,
        "distinct"   : distinct,
        "distinct_on": distinct_on,
        "similarity" : similarity,
        "between"    : between,
        "fuzzy_in"   : fuzzy_in,
    }


def _parse_in_list(raw: str, tok: _Tokens):
    """
    Parse  (v1, v2, v3)  from remaining tokens.
    Returns (list_of_floats, updated_tok).
    """
    m = re.search(r'\(([^)]+)\)', raw)
    if not m:
        raise ParseError(
            "IN requires a parenthesised list: IN (v1, v2, v3)"
        )
    try:
        values = [float(v.strip()) for v in m.group(1).split(",")]
    except ValueError:
        raise ParseError("IN list must contain numeric values only.")

    # Advance token stream past the closing paren
    full = " ".join(tok.remaining_raw())
    end  = full.index(")") + 1
    rest = full[end:].strip().split()
    # Rebuild tok from remaining text after the IN(...) list
    tok2 = _Tokens(rest)
    return values, tok2


def _parse_where_conditions(tok: _Tokens) -> list[dict]:
    """
    Parse chained IS conditions:
      <col> IS [<hedge>] <term> [AND|OR <col> IS [<hedge>] <term>] ...
    """
    conditions = []
    logic = "AND"

    stop_words = {"THRESHOLD", "TOP", "ON", "SIMILARITY"}

    while tok.has():
        if tok.peek() in stop_words:
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
        raise ParseError("WHERE clause produced no conditions.")
    return conditions


# ─────────────────────────────────────────────────────────────────────────────
# JOIN
# ─────────────────────────────────────────────────────────────────────────────

def _parse_join(tok: _Tokens) -> dict:
    """
    JOIN <left_table> WITH <right_table>
         ON <col>
         [TYPE inner | left | right | full]
         [TOLERANCE <float>]
         [THRESHOLD <float>]
         [TOP <int>]

    Examples
    --------
    JOIN employees WITH departments ON dept
    JOIN employees WITH departments ON dept TYPE left TOLERANCE 0.1
    """
    tok.expect("JOIN")

    left_table = tok.consume_raw()
    tok.expect("WITH")
    right_table = tok.consume_raw()
    tok.expect("ON")
    on_col = tok.consume_raw()

    join_type = "inner"
    tolerance = 0.0
    threshold = 0.0
    top_k     = None

    while tok.has():
        kw = tok.peek()
        if kw == "TYPE":
            tok.consume()
            jt = tok.consume_raw().lower()
            if jt not in KNOWN_JOIN_TYPES:
                raise ParseError(
                    f"Unknown join type {jt!r}. "
                    f"Valid: {sorted(KNOWN_JOIN_TYPES)}"
                )
            join_type = jt
        elif kw == "TOLERANCE":
            tok.consume()
            tolerance = tok.consume_float("TOLERANCE")
        elif kw == "THRESHOLD":
            tok.consume()
            threshold = tok.consume_float("THRESHOLD")
        elif kw == "TOP":
            tok.consume()
            top_k = tok.consume_int("TOP")
        else:
            raise ParseError(
                f"Unexpected token in JOIN: {tok.peek_raw()!r}. "
                "Expected TYPE, TOLERANCE, THRESHOLD, or TOP."
            )

    return {
        "cmd"        : "join",
        "left_table" : left_table,
        "right_table": right_table,
        "on"         : on_col,
        "join_type"  : join_type,
        "tolerance"  : tolerance,
        "threshold"  : threshold,
        "top_k"      : top_k,
    }


# ─────────────────────────────────────────────────────────────────────────────
# GROUPBY
# ─────────────────────────────────────────────────────────────────────────────

def _parse_groupby(tok: _Tokens) -> dict:
    """
    GROUPBY <table> ON <col>
            TERMS <term1>, <term2>, ...
            [AGGREGATE <agg_col> FUNCS <func1>, <func2>, ...]
            [MIN_MEMBERSHIP <float>]

    Alias: GROUP BY  (two tokens) also accepted.

    Examples
    --------
    GROUPBY employees ON age TERMS young, middle, senior
    GROUPBY employees ON age TERMS young, senior AGGREGATE salary FUNCS weighted_avg, fuzzy_count
    """
    tok.expect("GROUP", "GROUPBY")
    # Handle "GROUP BY" as two-token alias
    if tok.peek() == "BY":
        tok.consume()

    table = tok.consume_raw()
    tok.expect("ON")
    col = tok.consume_raw()
    tok.expect("TERMS")

    # Read comma-separated term names until next keyword
    stop = {"AGGREGATE", "MIN_MEMBERSHIP"}
    terms = _read_csv_list(tok, stop)
    if not terms:
        raise ParseError("GROUPBY requires at least one TERM name.")

    agg_col   = None
    agg_funcs = ["fuzzy_count", "weighted_avg", "avg", "min", "max"]
    min_mem   = 0.0

    while tok.has():
        kw = tok.peek()
        if kw == "AGGREGATE":
            tok.consume()
            agg_col = tok.consume_raw()
            tok.expect("FUNCS")
            agg_funcs = _read_csv_list(tok, set())
            if not agg_funcs:
                raise ParseError("FUNCS requires at least one function name.")
            bad = set(agg_funcs) - KNOWN_AGG_FUNCS
            if bad:
                raise ParseError(
                    f"Unknown aggregate function(s): {bad}. "
                    f"Valid: {sorted(KNOWN_AGG_FUNCS)}"
                )
        elif kw == "MIN_MEMBERSHIP":
            tok.consume()
            min_mem = tok.consume_float("MIN_MEMBERSHIP")
        else:
            raise ParseError(
                f"Unexpected token in GROUPBY: {tok.peek_raw()!r}."
            )

    return {
        "cmd"      : "groupby",
        "table"    : table,
        "col"      : col,
        "terms"    : terms,
        "agg_col"  : agg_col,
        "agg_funcs": agg_funcs,
        "min_mem"  : min_mem,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SET OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

def _parse_set_op(tok: _Tokens) -> dict:
    """
    UNION      <table1>  AND  <table2>  ON  <key>
    INTERSECT  <table1>  AND  <table2>  ON  <key>
    EXCEPT     <table1>  FROM <table2>  ON  <key>

    Both tables must already have been queried or loaded.
    The engine stores the last query result under the table name
    with suffix _result (e.g. employees_result).

    Examples
    --------
    UNION employees AND contractors ON id
    INTERSECT senior_result AND highsalary_result ON id
    EXCEPT fulltime FROM parttime ON id
    """
    op = tok.consume_upper()    # UNION | INTERSECT | EXCEPT

    table_a = tok.consume_raw()

    # EXCEPT uses FROM, others use AND
    if op == "EXCEPT":
        tok.expect("FROM")
    else:
        tok.expect("AND")

    table_b = tok.consume_raw()
    tok.expect("ON")
    key = tok.consume_raw()

    threshold = 0.0
    if tok.has() and tok.peek() == "THRESHOLD":
        tok.consume()
        threshold = tok.consume_float("THRESHOLD")

    return {
        "cmd"      : "set_op",
        "op"       : op.lower(),     # union | intersect | except
        "table_a"  : table_a,
        "table_b"  : table_b,
        "key"      : key,
        "threshold": threshold,
    }


# ─────────────────────────────────────────────────────────────────────────────
# COMPARE  (the viva proof command)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_compare(tok: _Tokens) -> dict:
    """
    COMPARE <table>
            FUZZY   <col> IS [<hedge>] <term> [AND|OR ...] ...
            AGAINST <col> <op> <value>        [AND      ...] ...
            [THRESHOLD <float>]

    Runs a fuzzy query and an equivalent classical SQL query side by side,
    then reports what each found, what only fuzzy found, and why it matters.

    Classical operators: <  <=  >  >=  ==  !=

    Examples
    --------
    COMPARE employees FUZZY age IS young AND salary IS medium AGAINST age <= 35 AND salary >= 40000
    COMPARE employees FUZZY experience IS senior AGAINST experience >= 15 THRESHOLD 0.3
    """
    tok.expect("COMPARE")
    table = tok.consume_raw()
    tok.expect("FUZZY")

    # Parse fuzzy conditions up to AGAINST
    fuzzy_conditions = _parse_conditions_until(tok, stop_word="AGAINST")

    tok.expect("AGAINST")

    # Parse classical conditions:  col op value [AND col op value ...]
    classical_conditions = _parse_classical_conditions(tok)

    threshold = 0.0
    if tok.has() and tok.peek() == "THRESHOLD":
        tok.consume()
        threshold = tok.consume_float("THRESHOLD")

    return {
        "cmd"                 : "compare",
        "table"               : table,
        "fuzzy_conditions"    : fuzzy_conditions,
        "classical_conditions": classical_conditions,
        "threshold"           : threshold,
    }


def _parse_conditions_until(tok: _Tokens, stop_word: str) -> list[dict]:
    """Parse IS conditions until a stop keyword is encountered."""
    conditions = []
    logic = "AND"
    stop  = stop_word.upper()

    while tok.has() and tok.peek() != stop:
        col = tok.consume_raw()
        tok.expect("IS")

        hedge = None
        if tok.has() and tok.peek() and tok.peek().lower() in KNOWN_HEDGES:
            hedge = tok.consume_raw().lower()

        if not tok.has():
            raise ParseError(f"Missing term after 'IS' for column {col!r}.")
        term = tok.consume_raw()

        conditions.append({
            "col"  : col,
            "hedge": hedge,
            "term" : term,
            "logic": logic,
        })

        if tok.has() and tok.peek() in ("AND", "OR") and tok.peek(1) != stop:
            logic = tok.consume_upper()
        elif tok.has() and tok.peek() == "AND" and tok.peek(1) == stop:
            tok.consume()   # consume the AND before AGAINST
            break
        else:
            break

    return conditions


def _parse_classical_conditions(tok: _Tokens) -> dict:
    """
    Parse:  col op value [AND col op value ...]
    Returns: { col: (op, value) }
    """
    conditions = {}
    stop = {"THRESHOLD"}

    while tok.has() and tok.peek() not in stop:
        col = tok.consume_raw()
        op  = tok.consume_raw()

        # Normalize = to ==
        if op == "=":
            op = "=="

        if op not in CLASSICAL_OPS:
            raise ParseError(
                f"Unknown operator {op!r} in AGAINST clause. "
                f"Valid: {sorted(CLASSICAL_OPS)}"
            )

        val_raw = tok.consume_raw()
        try:
            val = float(val_raw)
        except ValueError:
            val = val_raw  # keep as string for text columns

        conditions[col] = (op, val)

        if tok.has() and tok.peek() == "AND":
            tok.consume()

    if not conditions:
        raise ParseError(
            "AGAINST clause requires at least one condition: col op value."
        )
    return conditions


# ─────────────────────────────────────────────────────────────────────────────
# ANALYZE
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# DEFUZZ
# ─────────────────────────────────────────────────────────────────────────────

def _parse_defuzz(tok: _Tokens) -> dict:
    """DEFUZZ <table>.<col> IS <term> METHOD <method>"""
    tok.expect("DEFUZZ")
    ref = tok.consume_raw()
    if "." not in ref:
        raise ParseError(f"Expected <table>.<column>, got {ref!r}.")
    table, col = ref.split(".", 1)
    tok.expect("IS")
    term = tok.consume_raw()

    method = "all"
    if tok.has() and tok.peek() == "METHOD":
        tok.consume()
        method = tok.consume_raw().lower()
        if method not in KNOWN_DEFUZZ_METHODS and method != "all":
            raise ParseError(
                f"Unknown defuzz method {method!r}. "
                f"Valid: all | {' | '.join(sorted(KNOWN_DEFUZZ_METHODS))}"
            )

    return {
        "cmd"   : "defuzz",
        "table" : table,
        "col"   : col,
        "term"  : term,
        "method": method,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SUGGEST
# ─────────────────────────────────────────────────────────────────────────────

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
        n = tok.consume_int("N")
    return {"cmd": "suggest", "table": table, "col": col, "n": n}


# ─────────────────────────────────────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────────────────────────────────────

def _parse_plot(tok: _Tokens) -> dict:
    """PLOT <table>.<col>"""
    tok.expect("PLOT")
    ref = tok.consume_raw()
    if "." not in ref:
        raise ParseError(f"Expected <table>.<column>, got {ref!r}.")
    table, col = ref.split(".", 1)
    return {"cmd": "plot", "table": table, "col": col}


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_csv_list(tok: _Tokens, stop_words: set) -> list[str]:
    """
    Read a comma-separated list of identifiers from the token stream.
    Items may be split by spaces or commas.
    Stops when a keyword in stop_words is encountered.
    """
    items = []
    raw   = " ".join(tok.remaining_raw())

    # Find position of first stop word
    upper_raw = raw.upper()
    stop_pos  = len(raw)
    for sw in stop_words:
        idx = re.search(r'\b' + sw + r'\b', upper_raw)
        if idx:
            stop_pos = min(stop_pos, idx.start())

    list_part = raw[:stop_pos].strip()
    rest_part = raw[stop_pos:].strip()

    for item in re.split(r'[\s,]+', list_part):
        if item:
            items.append(item.strip())

    # Rebuild tok from remaining text after the list
    rest_tokens = rest_part.split() if rest_part else []
    tok._t = tok._t[:tok._i] + rest_tokens
    # Fast forward past what we consumed
    tok._i = tok._i + len(list_part.split())   # approximate

    # Actually, simpler: drain and rebuild
    tok._t = rest_tokens
    tok._i = 0

    return items