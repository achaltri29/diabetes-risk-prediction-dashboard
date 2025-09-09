import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib

# Load dataset
df = pd.read_csv("patient_features_with_outcome.csv")

# Encode categorical variable "Sex"
if "Sex" in df.columns:
    df["Sex"] = LabelEncoder().fit_transform(df["Sex"])

# Features & target
X = df.drop(columns=["PtID", "Outcome"])
y = df["Outcome"]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Train logistic regression
log_reg = LogisticRegression(max_iter=1000, class_weight="balanced")
log_reg.fit(X_train, y_train)

# Evaluate
y_pred = log_reg.predict(X_test)
y_proba = log_reg.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model & scaler
joblib.dump(log_reg, "logistic_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("✅ Model and scaler saved.")
