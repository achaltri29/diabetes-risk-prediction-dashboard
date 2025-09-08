## AI‑Driven Patient Deterioration Risk Dashboard (Welldoc)

Production demo: https://diabetes-risk-prediction-dashboard.streamlit.app

### Overview
This project is a clinician‑focused Streamlit dashboard that surfaces patient‑level and cohort‑level deterioration (mortality) risk. It loads local CSV files as the "database" and renders a clean 2‑page experience:
- Patients (Cohort Overview + Patient Detail)
- Model Metrics (static performance images)

Key capabilities:
- Cohort table with filtering (risk, age, sex, outcome) and PtID search
- Percent‑formatted, color‑coded Mortality Rate and Severity badges (Low/Moderate/High)
- AUROC, confusion matrix, accuracy, precision, recall
- Risk score distribution plot
- Patient detail snapshot, editable clinical notes, daily CGM trend, and simple explainability (if model/scaler provided)
- Model Metrics page shows four pre‑rendered images (PNG)

### Project Structure
```
WELLDOC/
  streamlit_app.py
  dashboard_utils.py
  patient_features_with_outcome.csv
  test_predictions.csv
  daily_mean_glucose_with_outcome.csv
  logistic_model.pkl          (optional)
  scaler.pkl                  (optional)
  calibration_curve.png       (Model Metrics image)
  confusion_matrix.png        (Model Metrics image)
  roc_curve.png               (Model Metrics image)
  precision_recall_curve.png  (Model Metrics image)
  README.md
```

### Requirements
- Python 3.10+ (tested with 3.13)
- Streamlit 1.49+
- pandas, numpy, scikit‑learn, plotly, seaborn, matplotlib

### Quickstart (Local)
1) Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) Install dependencies:
```bash
pip install --upgrade pip
pip install streamlit plotly scikit-learn seaborn matplotlib pandas numpy
```

3) Make sure all required CSV/PNG files are present in the same folder as `streamlit_app.py` (see structure above). Model files are optional.

4) Run the app:
```bash
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`.

### Data Sources (loaded from ./ on startup)
- `./patient_features_with_outcome.csv`
- `./test_predictions.csv`
- `./daily_mean_glucose_with_outcome.csv`

Optional artifacts for explainability (linear model):
- `./logistic_model.pkl`
- `./scaler.pkl`

Model Metrics images (PNG):
- `./calibration_curve.png`
- `./confusion_matrix.png`
- `./roc_curve.png`
- `./precision_recall_curve.png`

All paths are relative to the project root; no upload widgets or file choosers are used.

### Using the Dashboard
- Sidebar
  - Choose page: Patients or Model Performance (titled "Model Metrics" in earlier versions)
  - Filters: Predicted risk range, age range, sex, outcome, PtID search
- Cohort Overview
  - Table columns: PatientId, Age, Sex, Outcome, Mortality Rate (%), Severity
  - Mortality Rate and Severity are color‑coded (green <30%, yellow 30–<90%, red ≥90%)
  - Plots: risk score distribution; performance metrics shown above
  - Download filtered cohort as CSV
- Patient Detail
  - Select PtID to view features snapshot, Mortality Rate badge, high‑risk alert (≥90%)
  - Daily CGM trend with hypo/hyper highlighting (if data available)
  - Editable “Recommended Next Steps” notes
- Model Metrics
  - Displays the four static PNG images only; no inputs

### Notes on Explainability
If `logistic_model.pkl` (linear model with `coef_`) and `scaler.pkl` are present, the app:
- Shows a global feature importance bar chart (by absolute coefficient)
- Estimates per‑patient top contributing features by `coef * value` (post‑scaling)

### Troubleshooting
- If tables/plots are empty, verify CSV files exist and have the required columns: `PtID, Age, Sex, Outcome, PredictedProbability` (plus feature columns like `mean_glucose`).
- If images don’t appear on Model Metrics: confirm the `.png` files exist and match the filenames above.
- If you changed file names or locations, update the hardcoded `"./"` paths accordingly.

### Deploying on Streamlit Community Cloud
1) Push this folder to a public GitHub repo.
2) Create a new Streamlit app in Streamlit Community Cloud and point it to `streamlit_app.py`.
3) Add the CSV/PNG/PKL assets to the repo (or to app Secrets/Storage as appropriate).

### License
For Hackwell/Welldoc hackathon demo purposes. Replace or add your chosen license here.


