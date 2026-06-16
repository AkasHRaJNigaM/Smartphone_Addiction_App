"""
Settings and configuration for Smartphone Addiction Analysis Dashboard.

Centralizes all app-wide settings including logging, Streamlit page config,
and matplotlib theme configuration.
"""

import os
import logging
import streamlit as st
from datetime import datetime
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# LOGGING CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════


def setup_logging() -> logging.Logger:
    """Configure and return the application logger.

    Returns:
        logging.Logger: Configured logger instance
    """
    LOG_DIR = "logs"
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    log_filename = os.path.join(
        LOG_DIR, f"addiction_app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    # Configure root logger
    logger = logging.getLogger("addiction_analyzer")
    logger.setLevel(logging.DEBUG)

    # File handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    logger.info("=" * 80)
    logger.info("Smartphone Addiction Analysis Dashboard Started")
    logger.info(f"Log file: {log_filename}")
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

DATA_DIR = Path("data")
LOG_DIR = Path("logs")
