import pandas as pd
import os
import glob

# 1. Clean the data directory
data_dir = "data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
else:
    for file in glob.glob(os.path.join(data_dir, "*.csv")):
        os.remove(file)
        print(f"Removed old file: {file}")

# 2. Candidates Data
candidates_data = {
    "id": ["c1", "c2", "c3", "c4", "c5"],
    "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
    "experience_yrs": [3, 7, 12, 8, 15],
    "aptitude_score": [85, 70, 95, 80, 60],
    "culture_fit": [90, 85, 78, 95, 50]
}
pd.DataFrame(candidates_data).to_csv(os.path.join(data_dir, "candidates.csv"), index=False)

# 3. Jobs Data (Must share a column with candidates for JOIN testing)
jobs_data = {
    "job_id": ["j1", "j2", "j3"],
    "title": ["Junior Dev", "Senior Dev", "Lead Architect"],
    "aptitude_score": [80, 95, 75]
}
pd.DataFrame(jobs_data).to_csv(os.path.join(data_dir, "jobs.csv"), index=False)

print("Successfully created candidates.csv and jobs.csv")