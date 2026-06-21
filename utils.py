"""
utils.py

- Utility functions for Smartphone Addiction Analysis Dashboard.
- Includes data loading, model training, and reusable analysis functions.
"""

import os
import logging
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import BorderlineSMOTE, SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
import streamlit as st

# NOTE: Fixing issues - Logistic Regression model training
# import statsmodels.api as sm
# from numpy.linalg import matrix_rank
# from numpy.linalg import LinAlgError

from scipy.stats import f_oneway, levene

from constants import (
    AGE_BINS,
    AGE_LABELS,
    DATASET_NAME,
    GB_HYPERPARAMS,
    RF_HYPERPARAMS,
    MODEL_FEATURES,
    LOGIT_THRESHOLD,
    SMOTE_HYPERPARAMS,
    KAGGLE_DATASET_ID,
    BORDERLINE_SMOTE_HYPERPARAMS,
)

from config import get_dataset_path

logger = logging.getLogger("addiction_analyzer")


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════


@st.cache_data(show_spinner="Loading dataset…")
def load_data() -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Load dataset directly from Kaggle via kagglehub.

    Returns:
        tuple: (df, nan_distances)
    """
    logger.info("Loading dataset from Kaggle...")

    try:
        dataset_path = get_dataset_path()

        csv_files = list(dataset_path.glob("*.csv"))

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {dataset_path}")

        csv_file = csv_files[0]

        logger.info(f"Dataset loaded from: {csv_file}")

        df = pd.read_csv(csv_file)

    except Exception as e:
        logger.exception("Failed to load dataset")

        st.error(f"""
❌ Failed to load dataset from Kaggle.

Error:
{e}
""")

        st.stop()

    cols = df.columns.tolist()

    df.drop(columns=cols[:2], inplace=True)

    logger.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Fix NaN addiction_level → Mild using Euclidean distance
    grouped = df.groupby(
        "addiction_level",
        dropna=False,
    ).mean(numeric_only=True)

    nan_prof = grouped.loc[np.nan]

    dists = {
        lvl: ((grouped.loc[lvl] - nan_prof) ** 2).sum()
        for lvl in ["Mild", "Moderate", "Severe"]
    }

    nan_count = df["addiction_level"].isna().sum()

    df["addiction_level"] = df["addiction_level"].fillna("Mild")

    logger.info(f"Imputed {nan_count} NaN values in 'addiction_level' to 'Mild'")

    # Feature engineering
    df["sum_components"] = (
        df["social_media_hours"] + df["gaming_hours"] + df["work_study_hours"]
    )

    df["age_group"] = pd.cut(
        df["age"],
        bins=AGE_BINS,
        labels=AGE_LABELS,
        include_lowest=True,
    )

    logger.info("Feature engineering completed")
    logger.info(f"Data loading complete. Processed {len(df)} records")

    return df, dists  # type: ignore


# NOTE: For LOCAL deployment
# @st.cache_data(show_spinner="Loading dataset…")
# def load_data() -> Tuple[pd.DataFrame, Dict[str, float]]:
#     """Load dataset from local data directory. Download from Kaggle if missing.

#     Returns:
#         tuple: (df, nan_distances) where:
#             - df: Preprocessed DataFrame with 7500 rows
#             - nan_distances: Dict mapping addiction levels to Euclidean distances
#     """
#     logger.info("Loading dataset...")

#     # Create local data directory (cross-platform)
#     data_dir = Path("data")
#     data_dir.mkdir(exist_ok=True)

#     # Expected local file path
#     csv_file = data_dir / DATASET_NAME

#     if csv_file.exists():
#         logger.info(f"Loading from local cache: {csv_file}")
#         df = pd.read_csv(csv_file)
#     else:
#         logger.info("Dataset not found locally. Downloading from Kaggle...")
#         try:
#             import kagglehub
#             import shutil

#             # Set Kaggle cache to local data directory
#             os.environ["KAGGLEHUB_HOME"] = str(data_dir)
#             p = kagglehub.dataset_download(KAGGLE_DATASET_ID)
#             csv_files = list(Path(p).glob("*.csv"))
#             if not csv_files:
#                 raise FileNotFoundError(f"No CSV files found in {p}")

#             source_csv = csv_files[0]
#             logger.info(f"Downloaded dataset: {source_csv.name}")

#             # Copy to local data directory for future reuse
#             shutil.copy(source_csv, csv_file)
#             logger.info(f"Cached to: {csv_file}")
#             df = pd.read_csv(csv_file)
#         except Exception as e:
#             error_msg = str(e)
#             logger.error(f"Failed to load dataset: {error_msg}")
#             st.error(f"""❌ Failed to load dataset from Kaggle.

# **Error:** {error_msg}

# **To fix this:**
# 1. Ensure Kaggle API credentials are configured:
#    - Create `~/.kaggle/kaggle.json` with your API key
#    - Run: `kaggle config set -n path_leaf_max_depth -v 10`

# 2. Or manually download the dataset:
#    - Visit: https://www.kaggle.com/datasets/mrinal1704/smartphone-addiction-classification-dataset
#    - Extract to the `data/` folder in this project

# 3. Check the README for more setup instructions.
# """)
#             st.stop()

#     cols = df.columns.tolist()
#     df.drop(columns=cols[:2], inplace=True)
#     cols = cols[2:]
#     logger.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

#     # Fix NaN addiction_level → Mild using Euclidean distance
#     grouped = df.groupby("addiction_level", dropna=False).mean(numeric_only=True)
#     nan_prof = grouped.loc[np.nan]
#     dists = {
#         lvl: ((grouped.loc[lvl] - nan_prof) ** 2).sum()
#         for lvl in ["Mild", "Moderate", "Severe"]
#     }
#     nan_count = df["addiction_level"].isna().sum()
#     df["addiction_level"] = df["addiction_level"].fillna("Mild")
#     logger.info(f"Imputed {nan_count} NaN values in 'addiction_level' to 'Mild'")

#     # Feature engineering
#     df["sum_components"] = (
#         df["social_media_hours"] + df["gaming_hours"] + df["work_study_hours"]
#     )
#     df["age_group"] = pd.cut(
#         df["age"], bins=AGE_BINS, labels=AGE_LABELS, include_lowest=True
#     )
#     logger.info("Feature engineering completed")
#     logger.info(f"Data loading complete. Processed {len(df)} records")
#     return df, dists  # type: ignore


# ══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════

# NOTE: Fixing issues - Logistic Regression model training
# def diagnose_logit_matrix(X):
#     """
#     Diagnose common causes of singular matrix errors.
#     """
#     rank = matrix_rank(X.values)
#     cols = len(X.columns)

#     logger.info("=" * 60)
#     logger.info("LOGISTIC REGRESSION DIAGNOSTICS")
#     logger.info("=" * 60)

#     logger.info(f"Shape: {X.shape}")
#     logger.info(f"Rank : {rank}")
#     logger.info(f"Cols : {cols}")

#     if rank < cols:
#         logger.warning(f"Rank deficient matrix detected " f"(rank={rank}, cols={cols})")

#     constant_cols = [c for c in X.columns if X[c].nunique() <= 1]

#     if constant_cols:
#         logger.warning(f"Constant columns detected: {constant_cols}")

#     corr = X.corr().abs()

#     high_corr = []

#     for i in range(len(corr.columns)):
#         for j in range(i + 1, len(corr.columns)):
#             if corr.iloc[i, j] > 0.999:
#                 high_corr.append(
#                     (
#                         corr.columns[i],
#                         corr.columns[j],
#                         corr.iloc[i, j],
#                     )
#                 )

#     if high_corr:
#         logger.warning(f"Highly correlated column pairs: {high_corr}")

#     logger.info("=" * 60)

#     return {
#         "rank": rank,
#         "columns": cols,
#         "constant_cols": constant_cols,
#         "high_corr": high_corr,
#     }


@st.cache_resource(show_spinner="Training Gradient Boosting model…")
def train_model(df_hash: Any) -> Dict[str, Any]:  # type: ignore
    """Train Gradient Boosting, Random Forest, and Logistic Regression models.

    Args:
        df_hash: Hash value for cache invalidation (not used directly)

    Returns:
        dict: Model data including:
            - model: Trained Gradient Boosting model
            - features: List of feature names
            - metrics (acc, auc, rf_acc, rf_auc, lg_acc, lg_auc)
            - predictions and test data
    """
    logger.info("Starting model training...")
    df = load_data()[0]

    # Encode categorical variables
    cat_cols = [c for c in df.columns if df[c].dtype in ["object", "category"]]
    logger.info(f"Categorical columns identified: {cat_cols}")
    tr_df = pd.get_dummies(df, columns=cat_cols, prefix_sep="_", dtype="int")

    # Ensure all feature columns exist
    for f in MODEL_FEATURES:
        if f not in tr_df.columns:
            tr_df[f] = 0

    target = df["addicted_label"]
    X = tr_df[MODEL_FEATURES]
    y = target

    # Train/test split
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=42
    )
    logger.info(f"Train/test split: {len(x_train)}/{len(x_test)} samples")

    # ── GRADIENT BOOSTING ──
    logger.info("Applying Borderline-SMOTE for class balancing...")
    blsmote = BorderlineSMOTE(**BORDERLINE_SMOTE_HYPERPARAMS)
    x_tr_bl, y_tr_bl = blsmote.fit_resample(x_train, y_train)  # type: ignore
    logger.info(f"SMOTE result: {len(x_tr_bl)} samples (original: {len(x_train)})")

    logger.info("Training Gradient Boosting Classifier...")
    model = GradientBoostingClassifier(**GB_HYPERPARAMS)
    model.fit(x_tr_bl, y_tr_bl)
    y_pred = model.predict(x_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    logger.info(f"GB Model - Accuracy: {acc:.4f}, ROC-AUC: {auc:.4f}")

    # ── RANDOM FOREST ──
    logger.info("Training Random Forest for comparison...")
    rf = RandomForestClassifier(**RF_HYPERPARAMS)
    rf.fit(x_train, y_train)
    rf_pred = rf.predict(x_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_auc = roc_auc_score(y_test, rf_pred)
    logger.info(f"RF Model - Accuracy: {rf_acc:.4f}, ROC-AUC: {rf_auc:.4f}")

    # NOTE: ── LOGISTIC REGRESSION ── # Fixing issues - Logistic Regression model training
    logger.info("Training Logistic Regression for comparison...")

    smote = SMOTE(**SMOTE_HYPERPARAMS)
    x_tr_sm, y_tr_sm = smote.fit_resample(x_train, y_train)  # type: ignore
    try:
        smote = SMOTE(**SMOTE_HYPERPARAMS)

        x_tr_sm, y_tr_sm = smote.fit_resample(  # type: ignore
            x_train,
            y_train,
        )

        logit = LogisticRegression(
            max_iter=5000,
            random_state=42,
            solver="lbfgs",
        )

        logit.fit(
            x_tr_sm,
            y_tr_sm,
        )

        lp_bin = logit.predict(x_test)

        lp_prob = logit.predict_proba(x_test)[:, 1]

        lg_acc = accuracy_score(
            y_test,
            lp_bin,
        )

        lg_auc = roc_auc_score(
            y_test,
            lp_prob,
        )

        logger.info(f"Logit Model - Accuracy: {lg_acc:.4f}, " f"ROC-AUC: {lg_auc:.4f}")

    except Exception:
        logger.exception("Logistic regression training failed")

        lg_acc = None
        lg_auc = None

    # logit_X = sm.add_constant(x_tr_sm.copy(), has_constant="add")
    # diag_info = diagnose_logit_matrix(logit_X)

    # try:
    #     logit = sm.Logit(y_tr_sm, logit_X).fit(disp=False, maxiter=500)
    #     test_X = sm.add_constant(x_test.copy(), has_constant="add")

    #     lp = logit.predict(test_X)
    #     lp_bin = (lp > LOGIT_THRESHOLD).astype(int)
    #     lg_acc = accuracy_score(y_test, lp_bin)
    #     lg_auc = roc_auc_score(y_test, lp)

    #     logger.info(f"Logit Model - Accuracy: {lg_acc:.4f}, ROC-AUC: {lg_auc:.4f}")

    # except LinAlgError:
    #     logger.exception("Logistic regression failed with singular matrix")

    #     lg_acc = np.nan
    #     lg_auc = np.nan

    # except Exception:
    #     logger.exception("Unexpected logistic regression failure")

    #     lg_acc = np.nan
    #     lg_auc = np.nan

    logger.info("Model training completed successfully")
    logger.info(f"Best model: Gradient Boosting (Acc: {acc:.4f}, AUC: {auc:.4f})")

    return {
        "model": model,
        "features": MODEL_FEATURES,
        "x_test": x_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "report": report,
        "cm": cm,
        "acc": acc,
        "auc": auc,
        "rf_acc": rf_acc,
        "rf_auc": rf_auc,
        "lg_acc": lg_acc,
        "lg_auc": lg_auc,
        # "logit_rank": diag_info["rank"],
        # "logit_columns": diag_info["columns"],
        # "logit_constant_cols": diag_info["constant_cols"],
        # "logit_high_corr": diag_info["high_corr"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════


def run_anova_analysis(
    df: pd.DataFrame, groupby_col: str, group_vals: list
) -> pd.DataFrame:
    """Run one-way ANOVA and Levene's test for group comparisons.

    Args:
        df: DataFrame with data
        groupby_col: Column name to group by
        group_vals: List of group values to compare

    Returns:
        pd.DataFrame: Results with p-values and significance indicators
    """
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    rows = []

    for col in num_cols:
        grp_data = [df[df[groupby_col] == g][col].dropna() for g in group_vals]
        f_stat, p_f = f_oneway(*grp_data)
        _, p_l = levene(*grp_data)
        rows.append(
            {
                "Feature": col,
                "ANOVA p-value": round(p_f, 4),
                "Levene p-value": round(p_l, 4),
                "Significant (α=0.05)": "✅ Yes" if p_f < 0.05 else "❌ No",
            }
        )
    return pd.DataFrame(rows)
