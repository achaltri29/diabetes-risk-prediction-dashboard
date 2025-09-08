import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression

# --------------------------
# Load daily file
# --------------------------
df = pd.read_csv("daily_mean_glucose_with_outcome.csv")

# --------------------------
# Pick one patient
# --------------------------
ptid = "P001"   # change to the actual PtID you want
patient_df = df[df["PtID"] == ptid].sort_values("day").reset_index(drop=True)
print(df["PtID"].unique())