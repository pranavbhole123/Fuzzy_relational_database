import numpy as np
import pandas as pd
from main import FuzzyRDBEngine

# =====================================================================
# 1. Setup the Simulation Parameters
# =====================================================================
N_ITERATIONS = 10000
TRUE_BP = 119.0           # The patient's actual underlying Blood Pressure
NOISE_STD = 2.5           # Standard deviation of the sensor noise
CLASSICAL_BOUNDARY = 120  # Strict classical cutoff

print("======================================================")
print(f" MONTE CARLO SIMULATION (N={N_ITERATIONS})")
print(f" Proving Fuzzy SQL Robustness against Noisy Data")
print("======================================================\n")
print(f"Patient's True Blood Pressure : {TRUE_BP}")
print(f"Sensor Noise (Std Deviation)  : {NOISE_STD}")
print(f"Classical SQL Cutoff          : <= {CLASSICAL_BOUNDARY}")
print(f"Fuzzy SQL Definition          : 100% normal up to 115, drops to 0% at 125\n")

# =====================================================================
# 2. Generate the Noisy Data
# =====================================================================
np.random.seed(42)  # Set seed for reproducible results
# Generate 10,000 random readings centered around 119 with a standard deviation of 2.5
noisy_measurements = np.random.normal(TRUE_BP, NOISE_STD, N_ITERATIONS)

# Build a Pandas DataFrame to represent our database table
df = pd.DataFrame({
    "reading_id": range(N_ITERATIONS),
    "bp": noisy_measurements
})

# =====================================================================
# 3. Initialize the Fuzzy Engine
# =====================================================================
engine = FuzzyRDBEngine(":memory:")
engine.load_dataframe(df, "patients")

# Define what "normal" means mathematically using a Trapezoidal MF.
# a=0, b=0: The left side is completely flat at 1.0 (from 0 to 115).
# c=115, d=125: The right side ramps down from 1.0 at 115 to 0.0 at 125.
engine.define_term(
    table_name="patients",
    column="bp",
    term="normal",
    mf_type="trapezoidal",
    params={"a": 0, "b": 0, "c": 115, "d": 125},
    umin=0, umax=200
)

# =====================================================================
# 4. Evaluate Classical SQL
# =====================================================================
# Classical query: Keep ONLY rows where bp <= 120
classical_results = (df["bp"] <= CLASSICAL_BOUNDARY).astype(float)
classical_expected_value = classical_results.mean()
classical_variance = classical_results.var()

# =====================================================================
# 5. Evaluate Fuzzy SQL
# =====================================================================
# Fuzzy query: SELECT * FROM patients WHERE bp IS normal
fuzzy_df = engine.query(
    table_name="patients",
    conditions=[{"col": "bp", "hedge": None, "term": "normal", "logic": "AND"}],
    threshold=0.0  # Keep threshold at 0 to see the full mathematical distribution
)

# The engine sorts results by membership descending. 
# We sort back by reading_id to easily extract the math.
fuzzy_df = fuzzy_df.sort_values("reading_id")
fuzzy_memberships = fuzzy_df["_membership"]

fuzzy_expected_value = fuzzy_memberships.mean()
fuzzy_variance = fuzzy_memberships.var()

# =====================================================================
# 6. Print the Mathematical Proof
# =====================================================================
print("------------------------------------------------------")
print(" CLASSICAL SQL RESULTS (Binary 0 or 1)")
print("------------------------------------------------------")
print(f"Expected Inclusion Rate : {classical_expected_value:.4f}  (Passes {classical_expected_value*100:.1f}% of the time)")
print(f"Variance (Volatility)   : {classical_variance:.4f}")

print("\n------------------------------------------------------")
print(" FUZZY SQL RESULTS (Continuous 0.0 to 1.0)")
print("------------------------------------------------------")
print(f"Expected Membership     : {fuzzy_expected_value:.4f}  (Average Truth Value)")
print(f"Variance (Volatility)   : {fuzzy_variance:.4f}")

print("\n======================================================")
print(" CONCLUSION")
print("======================================================")
variance_reduction = ((classical_variance - fuzzy_variance) / classical_variance) * 100

print(f"By swapping Classical SQL for Fuzzy SQL, the variance (volatility) of the")
print(f"query result dropped by a massive {variance_reduction:.1f}%!")
print("\nProof: Classical SQL is dangerously brittle near boundaries. Because the")
print("patient's true BP (119) was close to the boundary (120), random sensor")
print(f"noise caused the database to randomly reject them {(1.0 - classical_expected_value)*100:.1f}% of the time.")
print("\nFuzzy SQL smoothly absorbed the noise, maintaining a highly stable and")
print("mathematically reliable expected membership score.")