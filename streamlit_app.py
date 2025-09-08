import os
import pickle
import pandas as pd
import streamlit as st
from dashboard_utils import (
    load_csv_cached,
    harmonize_prediction_frame,
    filter_cohort,
    compute_classification_metrics,
    plot_roc_curve,
    plot_confusion_matrix,
    plot_risk_distribution,
    get_linear_model_feature_importances,
    estimate_patient_contributions,
    plot_feature_importances,
    plot_patient_contributions,
    plot_daily_glucose,
    to_readable_explanations,
    get_feature_columns,
    format_percent,
    severity_from_prob,
    severity_color,
    risk_color,
)

# ------------------------------
# Page Configuration (FIXED)
# ------------------------------
st.set_page_config(
    page_title="AI-Driven Patient Deterioration Risk Dashboard",
    layout="wide",
)

# ------------------------------
# Data Loading
# ------------------------------
# This decorator caches the output of this function
@st.cache_data(show_spinner="Loading and preparing data...")
def load_all_data():
    """
    Loads all necessary CSV files and merges them into a primary dataframe.
    This logic now prioritizes the feature set and reliably merges predictions onto it.
    """
    pred_df = load_csv_cached("./test_predictions.csv")
    feat_df = load_csv_cached("./patient_features_with_outcome.csv")
    glucose_df = load_csv_cached("./daily_mean_glucose_with_outcome.csv")

    if feat_df is None and pred_df is None:
        return pd.DataFrame(), pd.DataFrame(), None, None # Return empty objects

    # Harmonize columns before merging
    if pred_df is not None:
        pred_df = harmonize_prediction_frame(pred_df)
    if feat_df is not None:
        feat_df = harmonize_prediction_frame(feat_df)

    merged = pd.DataFrame()
    if feat_df is not None:
        merged = feat_df.copy()
        if pred_df is not None:
            preds_to_merge = pred_df[['PtID', 'PredictedProbability']].drop_duplicates('PtID')
            if 'PredictedProbability' in merged.columns:
                merged = merged.drop(columns=['PredictedProbability'])
            merged = merged.merge(preds_to_merge, on='PtID', how='left')
    elif pred_df is not None:
        merged = pred_df

    if glucose_df is None:
        glucose_df = pd.DataFrame()

    # Load model artifacts inside the cached function
    model, scaler = None, None
    model_path = "./logistic_model.pkl"
    scaler_path = "./scaler.pkl"
    try:
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                model = pickle.load(f)
        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
    except Exception:
        # Don't show a warning here, handle it later if explainability is needed
        pass
        
    return merged, glucose_df, model, scaler

# --- Main Data Loading Execution ---
# This call is now cached and won't trigger warnings on every interaction
merged_df, glucose_df, model, scaler = load_all_data()

# ------------------------------
# UI Header and Sidebar
# ------------------------------
# FIXED: Removed the emoji from the title
st.title("AI-Driven Patient Deterioration Risk Dashboard")
st.caption("Displaying patient data and model predictions. Use the sidebar to filter the cohort.")

# FIXED: Removed the "About this dashboard" expander
# with st.expander("About this dashboard", expanded=False):
#     st.write(APP_DESCRIPTION)

# --- Sidebar Filters ---
st.sidebar.title("Menu")
menu = st.sidebar.radio("Navigation", ["Cohort Dashboard", "Model Performance"], label_visibility="collapsed")
st.sidebar.header("Cohort Filters")

if merged_df.empty:
    st.warning("No patient data found to display. Please check your CSV files.")
    st.stop()

# --- More robust filter widgets to prevent crashes on empty/NaN data ---
prob_min_val, prob_max_val = (0.0, 1.0)
if 'PredictedProbability' in merged_df and merged_df['PredictedProbability'].notna().any():
    prob_min_val = float(merged_df['PredictedProbability'].min())
    prob_max_val = float(merged_df['PredictedProbability'].max())
prob_range = st.sidebar.slider(
    "Predicted Risk Range", min_value=0.0, max_value=1.0,
    value=(prob_min_val, prob_max_val), step=0.01
)

age_min_val, age_max_val = (0, 100)
if 'Age' in merged_df and merged_df['Age'].notna().any():
    age_min_val = int(merged_df['Age'].min())
    age_max_val = int(merged_df['Age'].max())
age_range = st.sidebar.slider("Age Range", min_value=0, max_value=120, value=(age_min_val, age_max_val))

sex_options = sorted(merged_df["Sex"].dropna().unique().tolist()) if "Sex" in merged_df.columns else []
sex_filter = st.sidebar.multiselect("Sex", options=sex_options, default=sex_options)

outcome_options_map = {0: "Alive", 1: "Deteriorated"}
outcome_filter_str = st.sidebar.multiselect("Actual Outcome", options=list(outcome_options_map.values()), default=list(outcome_options_map.values()))
outcome_filter = [k for k, v in outcome_options_map.items() if v in outcome_filter_str]

ptid_query = st.sidebar.text_input("Search by Patient ID")

filtered_df = filter_cohort(
    merged_df, prob_range[0], prob_range[1], age_range, sex_filter, outcome_filter, ptid_query
)

# ==================================================================================================
# MAIN PAGE: COHORT DASHBOARD VIEW
# ==================================================================================================
if menu == "Cohort Dashboard":
    tab1, tab2 = st.tabs(["📈 Cohort Overview", "🧑‍⚕️ Patient Detail View"])

    with tab1:
        st.subheader(f"Cohort Overview ({len(filtered_df)} Patients Shown)")
        if 'PredictedProbability' not in filtered_df.columns:
            st.error("The `PredictedProbability` column is missing. Cannot display risk scores.")
            st.dataframe(filtered_df)
        else:
            display_df = pd.DataFrame()
            display_df['Patient ID'] = filtered_df['PtID']
            if 'Age' in filtered_df.columns: display_df['Age'] = filtered_df['Age'].round(0).astype('Int64')
            if 'Sex' in filtered_df.columns: display_df['Sex'] = filtered_df['Sex']
            if 'Outcome' in filtered_df.columns:
                display_df['Actual Outcome'] = filtered_df['Outcome'].map(outcome_options_map)

            def create_badge(value, background_color):
                return f"<div style='background-color:{background_color}; color:white; padding: 5px; border-radius: 5px; text-align: center; font-weight: bold;'>{value}</div>"

            html_df = display_df.copy()
            html_df['Risk Score'] = filtered_df['PredictedProbability'].apply(
                lambda p: create_badge(format_percent(p), risk_color(p))
            )
            html_df['Severity'] = filtered_df['PredictedProbability'].apply(
                lambda p: create_badge(severity_from_prob(p), severity_color(p))
            )
            st.write(html_df.to_html(escape=False, index=False), unsafe_allow_html=True)
            st.markdown("---")
            
            st.subheader("Risk Score Distribution")
            st.plotly_chart(plot_risk_distribution(filtered_df.dropna(subset=["PredictedProbability"])), use_container_width=True)
            
            st.download_button("Download Filtered Cohort Data", filtered_df.to_csv(index=False).encode("utf-8"), "filtered_cohort.csv", "text/csv")

    with tab2:
        st.subheader("Single Patient Deep Dive")
        patient_ids = filtered_df["PtID"].dropna().astype(str).unique().tolist()
        if not patient_ids:
            st.info("No patients available with the current filters. Please adjust filters in the sidebar.")
        else:
            selected_ptid = st.selectbox("Select a Patient ID to inspect", options=patient_ids)
            prow = filtered_df[filtered_df["PtID"].astype(str) == str(selected_ptid)].iloc[[0]]
            prob = prow["PredictedProbability"].iloc[0] if 'PredictedProbability' in prow.columns else None
            outcome = prow["Outcome"].iloc[0] if 'Outcome' in prow.columns else None

            st.markdown("##### Patient Snapshot")
            scols = st.columns(4)
            scols[0].metric("Age", f"{prow['Age'].iloc[0]:.0f}" if 'Age' in prow.columns and pd.notna(prow['Age'].iloc[0]) else "N/A")
            scols[1].metric("Sex", prow['Sex'].iloc[0] if 'Sex' in prow.columns else "N/A")
            scols[2].metric("Predicted Risk", format_percent(prob))
            scols[3].metric("Actual Outcome", outcome_options_map.get(outcome, "N/A"))

            if prob is not None and prob > 0.66:
                 st.error(f"**High Risk Alert:** This patient has a **{format_percent(prob)}** predicted risk of deterioration.")
            st.markdown("---")


            st.markdown("##### Daily Glucose Trend")
            fig = plot_daily_glucose(glucose_df, selected_ptid)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No daily glucose data available for this patient.")

# ==================================================================================================
# MAIN PAGE: MODEL PERFORMANCE VIEW
# ==================================================================================================
if menu == "Model Performance":
    st.subheader("Overall Model Performance Evaluation")
    st.write("These metrics evaluate the model's performance on the **entire unfiltered test set**.")
    if 'Outcome' not in merged_df.columns or 'PredictedProbability' not in merged_df.columns:
        st.warning("Cannot calculate performance metrics. 'Outcome' and/or 'PredictedProbability' are missing.")
    else:
        valid_data = merged_df.dropna(subset=['Outcome', 'PredictedProbability'])
        y_true = valid_data['Outcome'].astype(int).values
        y_score = valid_data['PredictedProbability'].values
        metrics = compute_classification_metrics(y_true, y_score, threshold=0.5)

        m_cols = st.columns(4)
        m_cols[0].metric("AUROC", f"{metrics.get('auroc', 0):.3f}", help="Area Under the ROC Curve.")
        m_cols[1].metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}", help="Overall percentage of correct predictions.")
        m_cols[2].metric("Precision", f"{metrics.get('precision', 0):.3f}", help="Of predicted positives, how many were correct.")
        m_cols[3].metric("Recall", f"{metrics.get('recall', 0):.3f}", help="Of actual positives, how many were caught.")
        st.markdown("---")

        p_cols = st.columns(2)
        with p_cols[0]:
            st.markdown("##### ROC Curve")
            st.plotly_chart(plot_roc_curve(y_true, y_score), use_container_width=True)
        with p_cols[1]:
            st.markdown("##### Confusion Matrix")
            st.plotly_chart(plot_confusion_matrix(y_true, y_score, threshold=0.5), use_container_width=True)

        p_cols2 = st.columns(2)
        with p_cols2[0]:
             if os.path.exists("./precision_recall_curve.png"):
                st.markdown("##### Precision-Recall Curve")
                st.image("./precision_recall_curve.png")
        with p_cols2[1]:
            if os.path.exists("./calibration_curve.png"):
                st.markdown("##### Model Calibration Curve")
                st.image("./calibration_curve.png")

