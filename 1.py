import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib

# ============================
# Step 1: Load daily data
# ============================
df = pd.read_csv("daily_mean_glucose_with_outcome.csv")

# Convert day to datetime if it's a date string
df["day"] = pd.to_datetime(df["day"], errors="coerce")

# Ensure glucose is numeric, invalids → NaN
df["GlucoseCGM"] = pd.to_numeric(df["GlucoseCGM"], errors="coerce")

# ============================
# Step 2: Feature Engineering (aggregate 90 days per patient)
# ============================
patients = []
for ptid, group in df.groupby("PtID"):
    group = group.sort_values("day").reset_index(drop=True)

    glucose = group["GlucoseCGM"].values
    days = np.arange(len(group)).reshape(-1, 1)

    # Remove NaN values
    mask = ~np.isnan(glucose)
    glucose = glucose[mask]
    days = days[mask]

    if len(glucose) == 0:
        continue

    features = {}
    features["PtID"] = ptid
    features["mean_glucose"] = np.mean(glucose)
    features["median_glucose"] = np.median(glucose)
    features["std_glucose"] = np.std(glucose)
    features["cv_glucose"] = (
        features["std_glucose"] / features["mean_glucose"]
        if features["mean_glucose"] != 0
        else 0
    )
    features["min_glucose"] = np.min(glucose)
    features["max_glucose"] = np.max(glucose)
    features["range_glucose"] = features["max_glucose"] - \
        features["min_glucose"]
    features["pct_hypo"] = np.mean(glucose < 70) * 100
    features["pct_hyper"] = np.mean(glucose > 180) * 100
    features["pct_severe_hyper"] = np.mean(glucose > 250) * 100

    # slope of glucose trend
    if len(glucose) > 1:
        reg = LinearRegression().fit(days, glucose)
        features["slope_glucose"] = reg.coef_[0]
    else:
        features["slope_glucose"] = 0

    # Demographics
    features["Age"] = group["Age"].iloc[0]
    features["Sex"] = group["Sex"].iloc[0]

    # Outcome (if present in daily file)
    if "Outcome" in group.columns:
        features["Outcome"] = group["Outcome"].iloc[0]

    patients.append(features)

patient_df = pd.DataFrame(patients)
print("Aggregated patient-level dataset shape:", patient_df.shape)
print(patient_df.head())

# ============================
# Step 3: Preprocess
# ============================
if "Sex" in patient_df.columns:
    le = LabelEncoder()
    patient_df["Sex"] = le.fit_transform(patient_df["Sex"])

patient_df = patient_df.dropna(subset=["Outcome"])

X = patient_df.drop(columns=["PtID", "Outcome"])
y = patient_df["Outcome"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================
# Step 4: Train/Test Split + Logistic Regression
# ============================
X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
    X_scaled,
    y,
    patient_df.index,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\nModel Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model & scaler
joblib.dump(model, "logistic_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("\n✅ Model and scaler saved.")

# ============================
# Step 5: Line-by-line predictions on test set
# ============================
results = patient_df.loc[test_idx, ["PtID", "Outcome"]].copy()
results["PredictedProbability"] = y_proba

results.to_csv("test_predictions.csv", index=False)
print("\n✅ Test predictions saved to test_predictions.csv")
print(results.head())