import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("daily_mean_glucose_with_outcome.csv")

# Function to compute per-patient features
def extract_features(patient_df):
    feats = {}
    glucose = patient_df["GlucoseCGM"].dropna().values
    days = np.arange(len(glucose)).reshape(-1, 1)

    if len(glucose) == 0:
        return pd.Series()

    # Central tendency
    feats["mean_glucose"] = np.mean(glucose)
    feats["median_glucose"] = np.median(glucose)

    # Variability
    feats["std_glucose"] = np.std(glucose)
    feats["cv_glucose"] = feats["std_glucose"] / (feats["mean_glucose"] + 1e-6)

    # Extremes
    feats["min_glucose"] = np.min(glucose)
    feats["max_glucose"] = np.max(glucose)
    feats["range_glucose"] = feats["max_glucose"] - feats["min_glucose"]

    # Risk metrics
    feats["pct_hypo"] = np.mean(glucose < 70)
    feats["pct_hyper"] = np.mean(glucose > 180)
    feats["pct_severe_hyper"] = np.mean(glucose > 250)

    # Temporal trend (slope)
    if len(glucose) > 1:
        model = LinearRegression().fit(days, glucose)
        feats["slope_glucose"] = model.coef_[0]
    else:
        feats["slope_glucose"] = 0

    # Demographics (take first row info since it's constant per patient)
    feats["Age"] = patient_df["Age"].iloc[0]
    feats["Sex"] = patient_df["Sex"].iloc[0]

    return pd.Series(feats)


# Group by patient and compute features
patient_features = df.groupby("PtID").apply(extract_features).reset_index()

# Get outcome for each patient
outcomes = df[["PtID", "Outcome"]].drop_duplicates()

# Merge features with outcome
final_df = patient_features.merge(outcomes, on="PtID")

# Ensure 'Outcome' is the last column
cols = [col for col in final_df.columns if col != 'Outcome'] + ['Outcome']
final_df = final_df[cols]

# Save the final dataframe
output_filename = "patient_features_with_outcome.csv"
final_df.to_csv(output_filename, index=False)

print(f"Successfully created {output_filename} with {len(final_df.columns)} columns.")
print(final_df.head())
