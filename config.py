"""
config.py

- Settings and configuration for Smartphone Addiction Analysis Dashboard.
- Centralizes all app-wide settings.
"""

import logging
from os import environ

import streamlit as st
from pathlib import Path

# NOTE: For LOCAL testing
# from dotenv import load_dotenv
# load_dotenv()

# NOTE: For DEPLOYMENT
import kagglehub

from constants import KAGGLE_DATASET_ID

KAGGLE_DATASET_ID = "jimarahman/smartphone-usage-and-addiction-analysis-dataset"


def get_dataset_path() -> Path:
    """Authenticate and download dataset (if needed)."""

    environ["KAGGLE_USERNAME"] = st.secrets["KAGGLE_USERNAME"]
    environ["KAGGLE_KEY"] = st.secrets["KAGGLE_KEY"]

    return Path(kagglehub.dataset_download(KAGGLE_DATASET_ID))


# ══════════════════════════════════════════════════════════════════════════════
# LOGGING CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════


def setup_logging() -> logging.Logger:
    """
    Configure and return the application logger.

    Streamlit Community Cloud does not provide persistent local storage,
    therefore logs are emitted to stdout/stderr and can be viewed from
    the Streamlit Cloud logs panel.
    """

    logger = logging.getLogger("addiction_analyzer")

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info("=" * 80)
    logger.info("Smartphone Addiction Analysis Dashboard Started")
    logger.info("Logging Mode: Console")
    logger.info("=" * 80)

    return logger


# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT PAGE CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════


def setup_streamlit_page():
    """Configure Streamlit page settings (layout, title, icon, etc.)."""
    st.set_page_config(
        page_title="Smartphone Addiction Analysis",
        page_icon="📱",
        layout="wide",
        initial_sidebar_state="expanded",
    )


# ══════════════════════════════════════════════════════════════════════════════
# MATPLOTLIB CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

MATPLOTLIB_CONFIG = {
    "figure.facecolor": "#0f1117",
    "axes.facecolor": "#141414",
    "axes.edgecolor": "#333",
    "axes.labelcolor": "#ccc",
    "xtick.color": "#999",
    "ytick.color": "#999",
    "text.color": "#e0e0e0",
    "grid.color": "#2a2a2a",
    "grid.alpha": 0.6,
    "legend.facecolor": "#1a1a1a",
    "legend.edgecolor": "#444",
}

# ══════════════════════════════════════════════════════════════════════════════
# DATA DIRECTORIES
# ══════════════════════════════════════════════════════════════════════════════

# NOTE: For LOCAL testing
# DATA_DIR = Path("data")
# LOG_DIR = Path("logs")
