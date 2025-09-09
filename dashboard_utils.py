from __future__ import annotations
import os
import pandas as pd
import numpy as np
from sklearn import metrics as sk_metrics
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Tuple


# Data loading and validation
REQUIRED_COLUMNS = [
    "PtID", "Age", "Sex", "Outcome", "PredictedProbability",
]

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize expected columns if common variants are present."""
    df = df.copy()
    column_map = {}
    variants = {
        "ptid": ["ptid", "patient_id", "patientid", "id"],
        "age": ["age"],
        "sex": ["sex", "gender"],
        "outcome": ["outcome", "label", "y_true", "actual"],
        "predictedprobability": [
            "predictedprobability", "prediction", "pred", "risk",
            "score", "y_pred_proba",
        ],
    }
    lower_cols = {c.lower().strip(): c for c in df.columns}
    for target, var_list in variants.items():
        for var in var_list:
            if var in lower_cols:
                canonical_name = ''.join(word.capitalize() for word in target.split())
                if canonical_name == "Ptid": canonical_name = "PtID"
                if canonical_name == "Predictedprobability": canonical_name = "PredictedProbability"
                column_map[lower_cols[var]] = canonical_name
                break
    df.rename(columns=column_map, inplace=True)
    return df

def load_csv_cached(path: str) -> Optional[pd.DataFrame]:
    """Load a CSV with basic cleaning; returns None if file is missing."""
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    df.columns = [c.strip() for c in df.columns]
    df = standardize_columns(df)
    return df

def harmonize_prediction_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the minimal required columns exist and types are reasonable."""
    df = df.copy()
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    if "PredictedProbability" in df.columns:
        df["PredictedProbability"] = pd.to_numeric(df["PredictedProbability"], errors="coerce").clip(0, 1)
    if "Outcome" in df.columns:
        outcome_map = {
            "yes": 1, "true": 1, "positive": 1, "1": 1, "1.0": 1, "dead": 1, "deceased": 1,
            "no": 0, "false": 0, "negative": 0, "0": 0, "0.0": 0, "alive": 0,
        }
        df["Outcome"] = df["Outcome"].astype(str).str.lower().map(outcome_map).fillna(df["Outcome"])
        df["Outcome"] = pd.to_numeric(df["Outcome"], errors="coerce").astype("Int64")
    if "Sex" in df.columns:
        sex_map = {"m": "Male", "male": "Male", "f": "Female", "female": "Female"}
        df["Sex"] = df["Sex"].astype(str).str.lower().map(sex_map).fillna(df["Sex"].astype(str).str.title())

    return df

def filter_cohort(
    df: pd.DataFrame,
    prob_min: float, prob_max: float,
    age_range: Tuple[float, float],
    sex_filter: List[str],
    outcome_filter: List[int],
    ptid_query: str,
) -> pd.DataFrame:
    """FIXED: Filter cohort dataframe more safely, checking for column existence."""
    data = df.copy()
    if "PredictedProbability" in data.columns and data["PredictedProbability"].notna().any():
        data = data[data["PredictedProbability"].between(prob_min, prob_max)]
    if "Age" in data.columns and data["Age"].notna().any():
        data = data[data["Age"].between(age_range[0], age_range[1])]
    if sex_filter and "Sex" in data.columns:
        data = data[data["Sex"].isin(sex_filter)]
    if outcome_filter and "Outcome" in data.columns and data["Outcome"].notna().any():
        data = data[data["Outcome"].isin(outcome_filter)]
    if ptid_query and "PtID" in data.columns:
        q = ptid_query.strip().lower()
        data = data[data["PtID"].astype(str).str.lower().str.contains(q)]
    return data


# Metrics & Plots
def compute_classification_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    try:
        metrics["auroc"] = float(sk_metrics.roc_auc_score(y_true, y_score))
    except (ValueError, TypeError):
        metrics["auroc"] = float("nan")
    y_pred = (y_score >= threshold).astype(int)
    metrics["accuracy"] = float(sk_metrics.accuracy_score(y_true, y_pred))
    metrics["precision"] = float(sk_metrics.precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(sk_metrics.recall_score(y_true, y_pred, zero_division=0))
    return metrics

def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> go.Figure:
    try:
        fpr, tpr, _ = sk_metrics.roc_curve(y_true, y_score)
        auc = sk_metrics.auc(fpr, tpr)
    except (ValueError, TypeError):
        fpr, tpr, auc = np.array([0, 1]), np.array([0, 1]), float("nan")

    fig = go.Figure(layout=dict(template="plotly_dark"))
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc:.3f})", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash", color="grey")))
    fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", margin=dict(l=20, r=20, t=5, b=20))
    return fig

def plot_confusion_matrix(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> go.Figure:
    y_pred = (y_score >= threshold).astype(int)
    cm = sk_metrics.confusion_matrix(y_true, y_pred)
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=["Predicted Negative", "Predicted Positive"], y=["Actual Negative", "Actual Positive"],
        colorscale="Blues", showscale=False, text=cm, texttemplate="%{text}",
    ), layout=dict(template="plotly_dark"))
    fig.update_layout(margin=dict(l=40, r=20, t=5, b=20))
    return fig

def plot_risk_distribution(df: pd.DataFrame) -> go.Figure:
    color_discrete_map = {"0": "#1f77b4", "1": "#ff7f0e", "nan": "#d3d3d3"}
    fig = px.histogram(
        df, x="PredictedProbability", color=df["Outcome"].astype(str) if "Outcome" in df else None,
        nbins=30, barmode="overlay", opacity=0.75,
        color_discrete_map=color_discrete_map, template="plotly_dark"
    )
    fig.update_layout(xaxis_title="Predicted Risk Score", yaxis_title="Number of Patients", margin=dict(l=20, r=20, t=5, b=20))
    return fig


def get_linear_model_feature_importances(model, feature_names: List[str]) -> Optional[pd.DataFrame]:
    if model is None or not hasattr(model, "coef_"): return None
    coefs = model.coef_[0] if np.ndim(model.coef_) == 2 else model.coef_
    if len(feature_names) != len(coefs): return None
    df = pd.DataFrame({"feature": feature_names, "coefficient": coefs})
    df["abs_importance"] = df["coefficient"].abs()
    return df.sort_values("abs_importance", ascending=True)

def estimate_patient_contributions(
    patient_row: pd.Series, feature_names: List[str], model=None, scaler=None, top_k: int = 8
) -> Optional[pd.DataFrame]:
    if model is None or not hasattr(model, "coef_"): return None
    values = patient_row.reindex(feature_names).fillna(0).values.reshape(1, -1)
    if scaler is not None and hasattr(scaler, "transform"):
        try: values = scaler.transform(values)
        except Exception: pass
    coefs = model.coef_[0] if np.ndim(model.coef_) == 2 else model.coef_
    if values.shape[1] != len(coefs): return None
    contributions = coefs * values.flatten()
    contrib_df = pd.DataFrame({
        "feature": feature_names,
        "contribution": contributions,
        "value": patient_row.reindex(feature_names).values,
    })
    contrib_df["abs_contribution"] = contrib_df["contribution"].abs()
    return contrib_df.sort_values("abs_contribution", ascending=True).tail(top_k)

def plot_feature_importances(imp_df: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        imp_df, x="coefficient", y="feature", orientation="h",
        color=np.where(imp_df["coefficient"] >= 0, "Increases Risk", "Decreases Risk"),
        color_discrete_map={"Increases Risk": "#ff7f0e", "Decreases Risk": "#1f77b4"},
        template="plotly_dark"
    )
    fig.update_layout(showlegend=False, margin=dict(l=120, r=20, t=5, b=20))
    return fig

def plot_patient_contributions(contrib_df: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        contrib_df, x="contribution", y="feature", orientation="h",
        color=np.where(contrib_df["contribution"] >= 0, "Increases Risk", "Decreases Risk"),
        color_discrete_map={"Increases Risk": "#ff7f0e", "Decreases Risk": "#1f77b4"},
        template="plotly_dark", custom_data=['value']
    )
    fig.update_traces(hovertemplate="%{y}<br>Value=%{customdata[0]:.2f}<br>Contribution=%{x:.3f}<extra></extra>")
    fig.update_layout(showlegend=False, margin=dict(l=120, r=20, t=5, b=20))
    return fig

# Daily glucose trends
def plot_daily_glucose(df_glucose: pd.DataFrame, ptid: str, hypo: float = 70.0, hyper: float = 180.0) -> Optional[go.Figure]:
    if df_glucose is None or df_glucose.empty or "PtID" not in df_glucose.columns: return None
    g = df_glucose[df_glucose["PtID"].astype(str) == str(ptid)].copy()
    if g.empty: return None
    glucose_col = next((c for c in ["GlucoseCGM", "Glucose", "Value"] if c in g.columns), None)
    if not glucose_col: return None
    date_col = next((c for c in ["day", "date", "Day", "Timestamp"] if c in g.columns), None)
    if date_col:
        g[date_col] = pd.to_datetime(g[date_col], errors="coerce")
        g = g.sort_values(date_col)
        x_vals = g[date_col]
    else: x_vals = g.index
    fig = go.Figure(layout=dict(template="plotly_dark"))
    fig.add_trace(go.Scatter(x=x_vals, y=g[glucose_col], mode="lines+markers", name="Glucose", line=dict(color="#1f77b4")))
    y_max = g[glucose_col].max() if g[glucose_col].notna().any() else hyper
    fig.add_hrect(y0=0, y1=hypo, fillcolor="rgba(255, 127, 14, 0.2)", line_width=0, annotation_text="Hypoglycemia", annotation_position="top left")
    fig.add_hrect(y0=hyper, y1=max(y_max, hyper + 10), fillcolor="rgba(255, 0, 0, 0.2)", line_width=0, annotation_text="Hyperglycemia", annotation_position="top left")
    fig.update_layout(xaxis_title="Time", yaxis_title="Glucose (mg/dL)", margin=dict(l=40, r=20, t=5, b=20))
    return fig

# Formatting Helpers 
def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return likely feature columns by excluding known metadata columns."""
    non_feature_cols = {"PtID", "Age", "Sex", "Outcome", "PredictedProbability"}
    return [c for c in df.columns if c not in non_feature_cols and pd.api.types.is_numeric_dtype(df[c])]

def to_readable_explanations(contrib_df: Optional[pd.DataFrame]) -> List[str]:
    """Produce simple human-friendly explanations based on signed contributions."""
    if contrib_df is None or contrib_df.empty:
        return ["No contribution data available."]
    explanations = []
    for _, r in contrib_df.sort_values("abs_contribution", ascending=False).iterrows():
        direction = "raised" if r["contribution"] >= 0 else "lowered"
        value_str = f"{r['value']:.1f}" if pd.notna(r['value']) else "N/A"
        explanations.append(f"**{r['feature']}** of **{value_str}** {direction} the risk score.")
    return explanations

def risk_color(prob: Optional[float]) -> str:
    if pd.isna(prob): return "#374151"
    if prob < 0.33: return "rgba(34, 197, 94, 0.7)"
    if prob < 0.66: return "rgba(245, 158, 11, 0.7)"
    return "rgba(239, 68, 68, 0.7)"

def format_percent(prob: Optional[float]) -> str:
    if pd.isna(prob): return "N/A"
    return f"{float(prob) * 100:.1f}%"

def severity_from_prob(prob: Optional[float]) -> str:
    if pd.isna(prob): return "Unknown"
    if prob < 0.33: return "Low"
    if prob < 0.66: return "Moderate"
    return "High"

def severity_color(prob: Optional[float]) -> str:
    if pd.isna(prob): return "#6b7280"
    if prob < 0.33: return "#22c55e"
    if prob < 0.66: return "#f59e0b"
    return "#ef4444"

