# Smartphone Addiction Analysis Dashboard

A **data analytics** dashboard for exploring **smartphone usage** patterns, identifying **behavioral indicators** of smartphone addiction, and evaluating **predictive machine learning models**.

## Overview

This project analyzes a dataset of `7,500` smartphone users to:

- Explore usage behavior through interactive visualizations.
- Investigate relationships between screen time, social media usage, gaming, stress, and addiction levels.
- Perform statistical analysis using ANOVA and Levene's tests.
- Build and compare machine learning models for addiction prediction.
- Provide an interactive interface for real-time addiction risk assessment.

## Dataset

- Source: [Smartphone Usage and Addiction Analysis Dataset](https://www.kaggle.com/datasets/jimarahman/smartphone-usage-and-addiction-analysis-dataset)
- Records: **7,500 users**
- Features include:
  - Daily screen time
  - Social media usage
  - Gaming hours
  - Work/study hours
  - Stress level
  - App opens per day
  - Academic/work impact
  - Addiction level and addiction label

## Features

### Exploratory Data Analysis
- Addiction distribution analysis
- Age-group segmentation
- Gender-based comparisons
- Screen time behavior analysis
- Correlation heatmaps

### Statistical Analysis
- One-way ANOVA
- Levene's Test for variance homogeneity
- Data-driven missing value imputation validation

### Machine Learning
- Gradient Boosting Classifier
- Random Forest Classifier
- Logistic Regression (benchmark model)
- Borderline-SMOTE for class imbalance handling

### Interactive Prediction
Users can input behavioral metrics and receive:
- Addiction classification
- Probability estimates
- Model confidence visualization

## Tech Stack

- NumPy
- Pandas
- Seaborn
- Matplotlib
- SciPy
- StatsModels
- Imbalanced-Learn
- Scikit-learn
- Streamlit

## Installation

```bash
git clone https://github.com/AkasHRaJNigaM/Smartphone_Addiction_App.git
cd Smartphone_Addiction_App
```

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the Application

```bash
streamlit run main.py
```

The application will be available locally at:

```bash
http://localhost:8501
```

## Project Structure

```text
*
├── data/
├── logs/
├── pages/
├── assets/
├── constants.py
├── config.py
├── utils.py
├── main.py
├── requirements.txt
└── README.md
```

## Modeling Approach

The primary prediction model uses:

- **Gradient Boosting** Classifier
- **Borderline-SMOTE** oversampling
- **Stratified train-test split**
- **ROC-AUC** and **Accuracy** for evaluation

> Additional benchmark models include Random Forest and Logistic Regression.

# Special thanks to @nirmit27 for preparing structure and frontend of the app. 
