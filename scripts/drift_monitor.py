import pandas as pd
from scipy.stats import ks_2samp

REFERENCE_DATA = "data/processed/reference_data.csv"
CURRENT_DATA = "data/processed/current_data.csv"

NUMERIC_COLUMNS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "AverageMonthlySpend",
]


def detect_drift(reference, current, threshold=0.05):
    results = []

    for column in NUMERIC_COLUMNS:
        if column not in reference.columns or column not in current.columns:
            continue

        statistic, p_value = ks_2samp(
            reference[column].dropna(),
            current[column].dropna()
        )

        results.append({
            "feature": column,
            "ks_statistic": round(float(statistic), 4),
            "p_value": round(float(p_value), 4),
            "drift_detected": p_value < threshold,
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    reference = pd.read_csv(REFERENCE_DATA)
    current = pd.read_csv(CURRENT_DATA)

    report = detect_drift(reference, current)

    print("\n=== DATA DRIFT REPORT ===")
    print(report.to_string(index=False))

    report.to_csv(
        "data/processed/drift_report.csv",
        index=False
    )

    print("\nDrift report saved to:")
    print("data/processed/drift_report.csv")