import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    classification_report
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# Step 1: Load Data, Model, and Scaler
# ==============================================================================
print("Loading data, model, and scaler...")

# Load the feature-engineered dataset
try:
    df = pd.read_csv("patient_features_with_outcome.csv")
except FileNotFoundError:
    print("Error: 'patient_features_with_outcome.csv' not found.")
    print("Please run the feature engineering script first to generate this file.")
    exit()

# Load the trained logistic regression model
try:
    model = joblib.load("logistic_model.pkl")
except FileNotFoundError:
    print("Error: 'logistic_model.pkl' not found.")
    print("Please run the model training script first to generate this file.")
    exit()

# Load the scaler
try:
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    print("Error: 'scaler.pkl' not found.")
    print("Please run the model training script first to generate this file.")
    exit()

print("✅ Assets loaded successfully.")

# ==============================================================================
# Step 2: Preprocess Data and Recreate Test Set
# ==============================================================================
print("\nPreprocessing data and recreating the test set...")

# Encode categorical 'Sex' variable
if "Sex" in df.columns and df['Sex'].dtype == 'object':
    le = LabelEncoder()
    df["Sex"] = le.fit_transform(df["Sex"])

# Define features (X) and target (y)
try:
    X = df.drop(columns=["PtID", "Outcome"])
    y = df["Outcome"]
    # Binarize outcome if it's a string like 'Alive'/'Deceased'
    if y.dtype == 'object':
        # Convert 'Alive' to 0 and anything else ('Deceased') to 1
        y = y.apply(lambda x: 1 if x.lower() != 'alive' else 0)
except KeyError as e:
    print(f"Error: Missing column in the dataset - {e}")
    exit()

# Scale features
X_scaled = scaler.transform(X)

# Recreate the *exact* same train/test split as in training
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print("✅ Test set recreated successfully.")

# ==============================================================================
# Step 3: Make Predictions on the Test Set
# ==============================================================================
print("\nMaking predictions on the test set...")
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
y_pred_class_str = model.predict(X_test)         # Predicted class labels (e.g., 'Alive', 'Deceased')

# --- FIX for TypeError ---
# Convert string predictions to numeric (0 or 1) to match the data type of y_test.
# This resolves the error by ensuring both true and predicted labels are integers.
y_pred_class = pd.Series(y_pred_class_str).apply(lambda x: 1 if x.lower() != 'alive' else 0)

print("✅ Predictions made and data types aligned.")

# ==============================================================================
# Step 4: Calculate and Display Evaluation Metrics
# ==============================================================================
print("\n--- Model Evaluation Metrics ---")

# --- AUROC (Area Under the Receiver Operating Characteristic Curve) ---
auroc = roc_auc_score(y_test, y_pred_proba)
print(f"AUROC: {auroc:.4f}")

# --- AUPRC (Area Under the Precision-Recall Curve) ---
auprc = average_precision_score(y_test, y_pred_proba)
print(f"AUPRC: {auprc:.4f}")

# --- F1 Score ---
f1 = f1_score(y_test, y_pred_class)
print(f"F1 Score: {f1:.4f}")

# --- Classification Report (includes precision, recall, f1-score) ---
print("\nClassification Report:")
print(classification_report(y_test, y_pred_class, target_names=["Alive (0)", "Deteriorated (1)"]))

# ==============================================================================
# Step 5: Generate and Save Visualizations
# ==============================================================================
print("\nGenerating and saving visualizations...")

# --- Plot 1: Confusion Matrix ---
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_class)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Alive', 'Predicted Deteriorated'],
            yticklabels=['Actual Alive', 'Actual Deteriorated'])
plt.title('Confusion Matrix', fontsize=16)
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.savefig("confusion_matrix.png")
print("✅ Confusion Matrix saved as confusion_matrix.png")

# --- Plot 2: ROC Curve ---
plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auroc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("roc_curve.png")
print("✅ ROC Curve saved as roc_curve.png")

# --- Plot 3: Precision-Recall Curve ---
plt.figure(figsize=(8, 6))
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {auprc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.grid(True)
plt.savefig("precision_recall_curve.png")
print("✅ Precision-Recall Curve saved as precision_recall_curve.png")

# --- Plot 4: Calibration Curve ---
plt.figure(figsize=(10, 8))
prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10, strategy='uniform')
plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Logistic Regression')
plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve')
plt.legend()
plt.grid(True)
plt.savefig("calibration_curve.png")
print("✅ Calibration Curve saved as calibration_curve.png")

print("\nAll tasks complete. Check the console output and the generated image files.")

