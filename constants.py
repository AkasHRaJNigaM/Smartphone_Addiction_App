"""
Constants for Smartphone Addiction Analysis Dashboard.

Holds all reusable constant values: colors, styles, feature names, bins, etc.
"""

# ══════════════════════════════════════════════════════════════════════════════
# COLOR SCHEME
# ══════════════════════════════════════════════════════════════════════════════

ACCENT = "#f0c040"
PALETTE = ["#f0c040", "#40a0f0", "#f04040", "#40f090"]

# ══════════════════════════════════════════════════════════════════════════════
# CSS STYLES
# ══════════════════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0d0d0d;
    border-right: 1px solid #2a2a2a;
}
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 0.95rem; }

/* Main background */
.stApp { background: #0f1117; color: #e8e8e8; }

/* Section headers */
.section-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2rem;
    color: #f0c040;
    border-left: 5px solid #f0c040;
    padding-left: 14px;
    margin-bottom: 6px;
}
.section-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: #888;
    margin-bottom: 28px;
    padding-left: 20px;
}

/* KPI cards */
.kpi-row { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 28px; }
.kpi-card {
    background: #1a1a2e;
    border: 1px solid #2e2e4a;
    border-radius: 12px;
    padding: 20px 28px;
    flex: 1;
    min-width: 150px;
    text-align: center;
}
.kpi-value { font-size: 2rem; font-weight: 800; color: #f0c040; }
.kpi-label { font-size: 0.78rem; color: #999; margin-top: 4px; font-family: 'Space Mono', monospace; }

/* Insight boxes */
.insight-box {
    background: #141414;
    border-left: 4px solid #f0c040;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin: 10px 0;
    font-size: 0.9rem;
    color: #ccc;
}
.insight-box strong { color: #f0c040; }

/* Metric comparison table */
.metric-table th { color: #f0c040 !important; }

/* Prediction result */
.pred-addicted {
    background: linear-gradient(135deg,#3a0000,#1a0000);
    border: 2px solid #ff4444;
    border-radius: 14px;
    padding: 28px;
    text-align: center;
}
.pred-not-addicted {
    background: linear-gradient(135deg,#003a00,#001a00);
    border: 2px solid #44ff88;
    border-radius: 14px;
    padding: 28px;
    text-align: center;
}
.pred-label { font-size: 2.4rem; font-weight: 800; }
.pred-prob  { font-size: 1rem; color: #aaa; margin-top: 8px; font-family: 'Space Mono', monospace; }

/* Matplotlib figures dark theme */
</style>
"""

# ══════════════════════════════════════════════════════════════════════════════
# MODEL FEATURES
# ══════════════════════════════════════════════════════════════════════════════

MODEL_FEATURES = [
    "daily_screen_time_hours",
    "social_media_hours",
    "stress_level_Low",
    "app_opens_per_day",
    "academic_work_impact_Yes",
    "gaming_hours",
    "work_study_hours",
]

# ══════════════════════════════════════════════════════════════════════════════
# AGE BINNING
# ══════════════════════════════════════════════════════════════════════════════

AGE_BINS = [18, 22, 27, 31, 35]
AGE_LABELS = ["18-22", "22-27", "27-31", "31-35"]

# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════

DATASET_NAME = "Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv"
KAGGLE_DATASET_ID = "jimarahman/smartphone-usage-and-addiction-analysis-dataset"
DATASET_ROWS = 7500

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR CONTENT
# ══════════════════════════════════════════════════════════════════════════════

SIDEBAR_FOOTER = """<small style='color:#555'>Dataset · 7 500 rows<br>Best Model · Gradient Boosting<br>Sampling · Borderline-SMOTE</small>"""

# ══════════════════════════════════════════════════════════════════════════════
# NAVIGATION PAGES
# ══════════════════════════════════════════════════════════════════════════════

PAGES = [
    "🏠 Overview",
    "📊 EDA & Findings",
    "🔬 Statistical Tests",
    "🤖 Model Report",
    "🎯 Predict Addiction",
]

# ══════════════════════════════════════════════════════════════════════════════
# MODEL HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

GB_HYPERPARAMS = {
    "learning_rate": 0.02,
    "n_estimators": 100,
    "criterion": "friedman_mse",
}

RF_HYPERPARAMS = {
    "random_state": 42,
    "n_estimators": 100,
    "criterion": "gini",
    "min_samples_split": 4,
    "n_jobs": -1,
}

SMOTE_HYPERPARAMS = {
    "sampling_strategy": "minority",
    "random_state": 42,
}

BORDERLINE_SMOTE_HYPERPARAMS = {
    "sampling_strategy": "minority",
    "kind": "borderline-1",
    "random_state": 42,
}

# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION PAGE DEFAULTS
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_DAILY_SCREEN = 6.0
DEFAULT_SOCIAL_MEDIA = 3.0
DEFAULT_GAMING = 1.0
DEFAULT_WORK_STUDY = 4.0
DEFAULT_APP_OPENS = 80
DEFAULT_STRESS_INDEX = 1
DEFAULT_ACADEMIC_IMPACT_INDEX = 1

# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION THRESHOLDS
# ══════════════════════════════════════════════════════════════════════════════

LOGIT_THRESHOLD = 0.4  # For logistic regression predictions
