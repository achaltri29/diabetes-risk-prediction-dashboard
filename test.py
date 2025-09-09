import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression

# Load daily glucose data
df = pd.read_csv("daily_mean_glucose_with_outcome.csv")
print("Available PtIDs:", df["PtID"].unique()[:10], "...")

# Pick one real patient for testing
ptid = "1.0_DLCP3"
patient_df = df[df["PtID"] == ptid].copy()

if patient_df.empty:
    raise ValueError(f"No data found for PtID={ptid}")

# Convert 'day' to datetime then numeric index
patient_df["day"] = pd.to_datetime(patient_df["day"], errors="coerce")
patient_df = patient_df.sort_values("day").reset_index(drop=True)
days = np.arange(len(patient_df)).reshape(-1, 1)
glucose = patient_df["GlucoseCGM"].values


# Feature engineering
features = {}
features["mean_glucose"] = np.mean(glucose)
features["median_glucose"] = np.median(glucose)
features["std_glucose"] = np.std(glucose)
features["cv_glucose"] = features["std_glucose"] 
features["mean_glucose"] if features["mean_glucose"] != 0 else 0
features["min_glucose"] = np.min(glucose)
features["max_glucose"] = np.max(glucose)
features["range_glucose"] = features["max_glucose"] - features["min_glucose"]

features["pct_hypo"] = np.mean(glucose < 70) * 100
features["pct_hyper"] = np.mean(glucose > 180) * 100
features["pct_severe_hyper"] = np.mean(glucose > 250) * 100

# Trend slope (day index vs glucose)
reg = LinearRegression().fit(days, glucose)
features["slope_glucose"] = reg.coef_[0]

# Demographics
features["Age"] = patient_df["Age"].iloc[0]
features["Sex"] = 1 if patient_df["Sex"].iloc[0] == "Female" else 0


# Load trained model & scaler
model = joblib.load("logistic_model.pkl")
scaler = joblib.load("scaler.pkl")

X_new = pd.DataFrame([features])
X_scaled = scaler.transform(X_new)

# Predict mortality probability
probability = model.predict_proba(X_scaled)[:, 1][0]
print(f"Patient {ptid} - Predicted 90-day mortality probability: {probability:.3f}")
