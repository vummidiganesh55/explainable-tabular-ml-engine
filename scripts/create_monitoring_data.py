import pandas as pd
from pathlib import Path

INPUT_FILE = "data/raw/telco_customer_churn.csv"

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
).fillna(0)

# Create the same features used by the ML project
df["AverageMonthlySpend"] = (
    df["TotalCharges"] / df["tenure"].replace(0, 1)
)

df["IsNewCustomer"] = (df["tenure"] <= 3).astype(int)

df["HasLongTermContract"] = (
    df["Contract"] != "Month-to-month"
).astype(int)

# Remove target and ID columns
monitoring_df = df.drop(
    columns=["Churn", "customerID"],
    errors="ignore"
)

# Split into reference and simulated current data
reference_data = monitoring_df.iloc[:3500].copy()
current_data = monitoring_df.iloc[3500:7000].copy()

reference_data.to_csv(
    OUTPUT_DIR / "reference_data.csv",
    index=False
)

current_data.to_csv(
    OUTPUT_DIR / "current_data.csv",
    index=False
)

print("Reference data:", reference_data.shape)
print("Current data:", current_data.shape)
print("Monitoring datasets created successfully.")