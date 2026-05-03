# Fuzzy Relational Database

A Python-based fuzzy relational database engine that extends traditional relational queries with fuzzy membership, linguistic terms, hedges, and fuzzy set operations.

## What this project is

This project implements a fuzzy relational database engine with:
- CSV loading and SQLite-backed table storage.
- Fuzzy membership function definitions for table columns.
- A query engine supporting fuzzy WHERE, BETWEEN, IN, JOIN, GROUP BY, DISTINCT, and set operations.
- Linguistic hedges such as `very`, `somewhat`, `not`, and more.
- Comparison between fuzzy queries and classical SQL queries.
- A command-line interactive shell with a custom Fuzzy Query Language (FQL).

## Folder structure

- `main.py` — public engine entry point used by the CLI.
- `cli/` — interactive shell, parser, and result display.
- `core/` — fuzzy logic engine, storage, membership functions, hedges, and operations.
- `data/` — sample CSV datasets.
- `tests/` — unit tests covering core logic and CLI behavior.

## Key supported features

- Fuzzy `SELECT` with linguistic terms: `WHERE age IS young`
- Logical composition: `AND`, `OR`, `WEIGHTED`
- Soft thresholds: `THRESHOLD 0.3`
- Fuzzy `BETWEEN` with adjustable softness
- Fuzzy `IN` membership with tolerance
- Fuzzy joins: `inner`, `left`, `right`, `full`
- Fuzzy `DISTINCT` to remove near-duplicates
- Fuzzy `GROUP BY` and aggregation
- Set operations: `UNION`, `INTERSECT`, `EXCEPT`
- Analysis commands: `ANALYZE`, `DEFUZZ`, `PLOT`
- `COMPARE` for fuzzy vs classical query comparison

## How to run the project

### 1. Install dependencies

Use your Python environment and install required packages. If a requirements file exists, install it like:

```powershell
python -m pip install -r requirements.txt
```

If no requirements file is present, install core dependencies manually:

```powershell
python -m pip install pandas numpy
```

### 2. Run the interactive shell

From the repository root:

```powershell
python cli\shell.py
```

For a persistent SQLite database file:

```powershell
python cli\shell.py --db fuzzy.db
```

To execute a script file of FQL commands:

```powershell
python cli\shell.py --script example.fql
```

## Why classical SQL is limited

Classical SQL requires exact predicates and crisp boundaries. Queries like `age > 30` or `salary BETWEEN 50000 AND 80000` treat every row as either in or out, which makes it hard to express imprecise concepts such as "young", "high salary", or "close matches".

Classical frameworks also struggle with partial matches, fuzzy joins, and graded relevance ranking. These limitations make it difficult to model real-world queries where terms are inherently vague.

## How fuzzy SQL helps

Fuzzy SQL in this project adds graded membership and linguistic terms to relational queries. It allows:

- soft matching of numeric values using terms like `young`, `high`, or `senior`
- hedges such as `very`, `somewhat`, and `not` to modify meaning
- thresholding to keep only rows that meet a minimum degree of relevance
- ranking results by membership score instead of binary inclusion
- fuzzy joins and set operations that tolerate near-matches

This makes queries more expressive and better suited for scenarios where exact boundaries are not sufficient.

## Basic CLI commands

- `LOAD <csv_path> AS <table>`
- `SHOW TABLES`
- `SHOW TERMS <table>`
- `DESCRIBE <table>`
- `DEFINE TERM <table>.<col> AS <term> USING <mf_type>(...)`
- `SELECT * FROM <table> WHERE <col> IS [<hedge>] <term> [AND|OR ...] [THRESHOLD <float>] [TOP <int>]`
- `SELECT * FROM <table> WHERE <col> BETWEEN <low> AND <high> [SOFTNESS <float>]`
- `SELECT DISTINCT * FROM <table> [ON <cols>] [SIMILARITY <float>]`
- `JOIN <left> WITH <right> ON <col> [TYPE inner|left|right|full]`
- `GROUPBY <table> ON <col> TERMS <term1>, <term2> [AGGREGATE <col> FUNCS <funcs>]`
- `UNION <table1> AND <table2> ON <key>`
- `INTERSECT <table1> AND <table2> ON <key>`
- `EXCEPT <table1> FROM <table2> ON <key>`
- `COMPARE <table> FUZZY ... AGAINST ...`

## Membership functions and hedges

Supported membership functions include:
- `triangular`
- `trapezoidal`
- `gaussian`
- `generalized_bell`
- `sigmoid`
- `s_shaped`
- `z_shaped`
- `pi_shaped`
- `singleton`

Supported linguistic hedges include:
- `very`, `extremely`, `indeed`, `plus`
- `somewhat`, `quite`, `slightly`, `minus`
- `more_or_less`, `not`, `not_very`

## Notes

- The engine stores loaded tables in SQLite, but an in-memory database is used by default.
- Every fuzzy query result includes a `_membership` score that ranks rows by how well they satisfy the query.
- The CLI parser accepts a flexible Fuzzy Query Language designed for experimentation and demonstration.
