"""
test_storage.py — Verify the Storage Layer works correctly
Run: python tests/test_storage.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.storage import StorageEngine


def separator(title):
    print(f"\n{'─'*50}")
    print(f"  {title}")
    print('─'*50)


# ── Test 1: In-memory DB, manual table creation ───────────────────────────
separator("TEST 1: Create table manually")

engine = StorageEngine(":memory:")
engine.create_table("students", {
    "name":   "TEXT",
    "age":    "REAL",
    "marks":  "REAL",
    "grade":  "TEXT"
})
print("Tables:", engine.list_tables())
print("Columns:", engine.get_columns("students"))


# ── Test 2: Insert and fetch rows ─────────────────────────────────────────
separator("TEST 2: Insert and fetch rows")

engine.insert_many("students", [
    {"name": "Alice", "age": 20, "marks": 88.5, "grade": "A"},
    {"name": "Bob",   "age": 22, "marks": 55.0, "grade": "C"},
    {"name": "Carol", "age": 21, "marks": 73.0, "grade": "B"},
])
rows = engine.fetch_all("students")
for r in rows:
    print(dict(r))

print(f"Row count: {engine.row_count('students')}")


# ── Test 3: Load CSV ──────────────────────────────────────────────────────
separator("TEST 3: Load CSV file")

csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "employees.csv")
df = engine.load_csv(csv_path, "employees")
print(df.head())
print(f"\nLoaded {engine.row_count('employees')} rows")
print("Numeric columns:", engine.get_numeric_columns("employees"))


# ── Test 4: Column stats ──────────────────────────────────────────────────
separator("TEST 4: Column stats (useful for MF design)")

stats = engine.column_stats("employees", "age")
print("Age stats:", stats)

mn, mx = engine.get_column_range("employees", "salary")
print(f"Salary range: {mn} → {mx}")


# ── Test 5: MF metadata storage ───────────────────────────────────────────
separator("TEST 5: Save & retrieve membership function definitions")

engine.save_mf("employees", "age",    "young",  "triangular",  [0,  25, 35])
engine.save_mf("employees", "age",    "middle", "triangular",  [30, 42, 55])
engine.save_mf("employees", "age",    "old",    "trapezoidal", [50, 60, 100, 100])
engine.save_mf("employees", "salary", "low",    "triangular",  [0,  0,  50000])
engine.save_mf("employees", "salary", "medium", "triangular",  [40000, 65000, 90000])
engine.save_mf("employees", "salary", "high",   "triangular",  [80000, 110000, 130000])

print("\nMFs for 'age':")
for mf in engine.get_mfs_for_column("employees", "age"):
    print(" ", mf)

print("\nAll MFs for 'employees' table:")
all_mfs = engine.get_all_mfs("employees")
for col, mfs in all_mfs.items():
    print(f"  {col}: {[m['term'] for m in mfs]}")


# ── Test 6: Raw SQL execution ──────────────────────────────────────────────
separator("TEST 6: Raw SQL (classical comparison module will use this)")

results = engine.execute_sql(
    "SELECT name, age, salary FROM employees WHERE age < 30 AND salary > 40000"
)
print("Classical SQL — age < 30 AND salary > 40000:")
for r in results:
    print(" ", dict(r))


# ── Test 7: DataFrame fetch ────────────────────────────────────────────────
separator("TEST 7: Fetch as DataFrame")

df = engine.fetch_as_dataframe("employees")
print(df[["name", "age", "salary"]].to_string(index=False))

engine.close()
print("\n✅ All storage layer tests passed!")
