# Fuzzy Relational Database Engine

A general-purpose fuzzy relational database engine that extends classical SQL
with fuzzy querying using linguistic terms and membership functions.

## Project Structure

```
fuzzy_rdb/
├── core/
│   ├── storage.py          # SQLite storage layer
│   ├── mf_registry.py      # Membership function registry
│   ├── fuzzifier.py        # Crisp → fuzzy degree conversion
│   ├── fql_parser.py       # Fuzzy Query Language parser
│   └── query_engine.py     # Fuzzy query processor + ranker
├── classical/
│   └── sql_engine.py       # Classical SQL query runner (for comparison)
├── ui/
│   └── app.py              # Streamlit UI
├── data/                   # Sample CSV datasets
├── tests/                  # Unit tests
├── main.py                 # Entry point / engine API
└── requirements.txt
```

## Fuzzy Query Language (FQL) Syntax

```sql
SELECT * FROM <table>
WHERE <attr> IS <term> [AND|OR <attr> IS <term> ...]
[THRESHOLD <0.0 - 1.0>]
[ORDER BY membership DESC|ASC]
```

## Example

```python
from main import FuzzyRDBEngine

engine = FuzzyRDBEngine()
engine.load_csv("data/employees.csv", table_name="employees")

engine.define_membership("employees", "age", "young",   "triangular", [0, 25, 35])
engine.define_membership("employees", "age", "old",     "trapezoidal", [55, 65, 100, 100])
engine.define_membership("employees", "salary", "high", "triangular", [80000, 100000, 120000])

results = engine.fuzzy_query(
    "SELECT * FROM employees WHERE age IS young AND salary IS high THRESHOLD 0.3"
)
```
