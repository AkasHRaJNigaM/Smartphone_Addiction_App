# -*- coding: utf-8 -*-
"""
main.py

- Application entrypoint for the Streamlit dashboard
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# ── Import configuration, constants, and utilities ────────────────────────────
from config import setup_logging, setup_streamlit_page, MATPLOTLIB_CONFIG
from constants import PAGES, ACCENT, PALETTE, CUSTOM_CSS, SIDEBAR_FOOTER
from utils import load_data, train_model, run_anova_analysis
from sklearn.metrics import RocCurveDisplay

# ── Setup application ──────────────────────────────────────────────────────────
setup_streamlit_page()
logger = setup_logging()
plt.rcParams.update(MATPLOTLIB_CONFIG)  # type: ignore

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════


# ── Load data and train models ─────────────────────────────────────────────────
df, nan_distances = load_data()
model_data = train_model(id(df))  # NOTE: Hash trick to cache once.

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAV
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## Smartphone Addiction Analysis 📱")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        PAGES,
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(SIDEBAR_FOOTER, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown(
        '<div class="section-title">Project Overview</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="section-sub">smartphone usage · addiction analysis · 7 500 users</div>',
        unsafe_allow_html=True,
    )

    addicted_pct = df["addicted_label"].mean() * 100
    avg_screen = df["daily_screen_time_hours"].mean()
    avg_social = df["social_media_hours"].mean()
    mild_pct = (df["addiction_level"] == "Mild").mean() * 100

    st.markdown(
        f"""
    <div class="kpi-row">
      <div class="kpi-card"><div class="kpi-value">{len(df):,}</div><div class="kpi-label">TOTAL USERS</div></div>
      <div class="kpi-card"><div class="kpi-value">{addicted_pct:.1f}%</div><div class="kpi-label">CLASSIFIED ADDICTED</div></div>
      <div class="kpi-card"><div class="kpi-value">{avg_screen:.1f}h</div><div class="kpi-label">AVG DAILY SCREEN TIME</div></div>
      <div class="kpi-card"><div class="kpi-value">{avg_social:.1f}h</div><div class="kpi-label">AVG SOCIAL MEDIA TIME</div></div>
      <div class="kpi-card"><div class="kpi-value">{mild_pct:.1f}%</div><div class="kpi-label">MILD ADDICTION LEVEL</div></div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Addiction Label Distribution")
        fig, ax = plt.subplots(figsize=(5, 5))
        vals = df["addicted_label"].value_counts()
        colors = [PALETTE[2], PALETTE[3]]
        wedges, texts, autotexts = ax.pie(  # type: ignore
            vals,
            labels=["Addicted (1)", "Not Addicted (0)"],
            autopct="%1.1f%%",
            startangle=90,
            colors=colors,
            wedgeprops=dict(width=0.55, edgecolor="#0f1117", linewidth=3),
        )
        for at in autotexts:
            at.set_color("#0f1117")
            at.set_fontweight("bold")
        ax.set_title("Addicted vs Not Addicted", color=ACCENT, fontweight="bold")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.markdown("#### Addiction Level Distribution")
        fig, ax = plt.subplots(figsize=(5, 5))
        order = ["Mild", "Moderate", "Severe"]
        counts = df["addiction_level"].value_counts().reindex(order)
        bars = ax.bar(
            order,
            counts,
            color=PALETTE[:3],
            edgecolor="#0f1117",
            linewidth=2,
            width=0.5,
        )
        for b in bars:
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 30,
                f"{int(b.get_height()):,}",
                ha="center",
                color="#ccc",
                fontsize=10,
            )
        ax.set_title("Counts by Addiction Level", color=ACCENT, fontweight="bold")
        ax.set_ylabel("Count")
        ax.grid(axis="y")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("---")
    st.markdown("### Key Findings at a Glance")
    findings = [
        (
            "🔍 Class Imbalance",
            "The dataset is skewed — addicted users make up the majority, "
            "requiring oversampling (Borderline-SMOTE) for robust modelling.",
        ),
        (
            "📅 Mild = Not Addicted",
            "Only Moderate & Severe addiction levels are labelled as addicted (label=1). "
            "Mild users are consistently label=0, confirmed by both the data and cross-tab analysis.",
        ),
        (
            "📅 NaN → Mild",
            "NaN values in <code>addiction_level</code> were imputed to 'Mild' via Euclidean "
            "distance comparison of group means — closest match confirmed statistically.",
        ),
        (
            "📈 Screen Time ≈ Weekend Time",
            "OLS regression shows R² = 0.93 between daily and weekend screen time — "
            "a habitual pattern, not casual usage.",
        ),
        (
            "🔗 Feature Overlap",
            "Moderate & Severe addiction classes share nearly identical feature distributions. "
            "This limits multiclass accuracy; binary classification (addicted / not) works far better.",
        ),
        (
            "🤖 Best Model",
            "Gradient Boosting with Borderline-SMOTE achieves the highest AUC, "
            f"outperforming Logistic Regression and Random Forest on all metrics.",
        ),
    ]
    for title, desc in findings:
        st.markdown(
            f'<div class="insight-box"><strong>{title}</strong><br>{desc}</div>',
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA & FINDINGS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA & Findings":
    st.markdown(
        '<div class="section-title">EDA & Findings</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="section-sub">exploratory data analysis · visual patterns · feature relationships</div>',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📌 Age & Addiction", "📌 Screen Time", "📌 Gender Analysis", "📌 Correlation"]
    )

    # ── Age & Addiction ──────────────────────────────────────────────────────
    with tab1:
        st.markdown("### Addiction Distribution Across Age Groups")
        fig, ax = plt.subplots(figsize=(9, 5))
        ct = pd.crosstab(df["age_group"], df["addiction_level"], normalize=True)
        ct.plot(
            kind="bar",
            ax=ax,
            color=PALETTE[:3],
            edgecolor="#0f1117",
            linewidth=1.5,
            width=0.7,
        )
        ax.set_title(
            "Proportion of Addiction Level per Age Group",
            color=ACCENT,
            fontweight="bold",
            fontsize=13,
        )
        ax.set_xlabel("Age Group")
        ax.set_ylabel("Proportion")
        ax.tick_params(axis="x", rotation=0)
        ax.grid(axis="y")
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown(
            """
        <div class="insight-box">
        <strong>U-Shaped Behavioural Pattern:</strong> The 31–35 age group shows the highest relative
        addiction proportion — career stability correlating with more discretionary screen time.
        The 22–27 group dips (active career-building phase), confirming a lifestyle-linked pattern.
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown("### Screen Time Mean & Variation by Age Group")
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        age_dist = df.groupby("age_group", observed=False)[num_cols].agg(
            ["mean", "std"]
        )

        show_col = st.selectbox("Choose feature to explore:", num_cols)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for i, stat in enumerate(["mean", "std"]):
            axes[i].plot(
                age_dist.index,
                age_dist[(show_col, stat)],
                color=PALETTE[i],
                marker="o",
                linewidth=2.5,
                markersize=7,
            )
            axes[i].set_title(
                f"{'Mean' if stat=='mean' else 'Std Dev'} of {show_col} by Age Group",
                color=ACCENT,
                fontweight="bold",
            )
            axes[i].grid(True)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── Screen Time ──────────────────────────────────────────────────────────
    with tab2:
        st.markdown("### Screen Time Overlap by Addiction Level")
        fig, ax = plt.subplots(figsize=(9, 5))
        colors_kde = {"Mild": PALETTE[3], "Moderate": PALETTE[1], "Severe": PALETTE[2]}
        for level, color in colors_kde.items():
            subset = df[df["addiction_level"] == level]["daily_screen_time_hours"]
            subset.plot.kde(ax=ax, label=level, color=color, linewidth=2.5)
            ax.fill_between(
                np.linspace(subset.min(), subset.max(), 200),
                [
                    ax.lines[-1].get_ydata()[  # type: ignore
                        np.argmin(np.abs(ax.lines[-1].get_xdata() - x))  # type: ignore
                    ]
                    for x in np.linspace(subset.min(), subset.max(), 200)
                ],  # type: ignore
                alpha=0.15,
                color=color,
            )
        ax.set_title(
            "KDE — Daily Screen Time by Addiction Level",
            color=ACCENT,
            fontweight="bold",
            fontsize=13,
        )
        ax.set_xlabel("Daily Screen Time (hours)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown(
            """
        <div class="insight-box">
        <strong>Feature Space Overlap:</strong> Moderate and Severe addiction levels have heavily
        overlapping screen time distributions — explaining why multiclass models plateau below 61% accuracy.
        Binary classification (addicted vs not) is the correct problem framing.
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown("### Weekend vs Daily Screen Time (OLS: R² = 0.93)")
        fig, ax = plt.subplots(figsize=(8, 5))
        sample = df.sample(500, random_state=1)
        ax.scatter(
            sample["weekend_screen_time"],
            sample["daily_screen_time_hours"],
            alpha=0.3,
            color=PALETTE[1],
            s=15,
        )
        m, b = np.polyfit(df["weekend_screen_time"], df["daily_screen_time_hours"], 1)
        xs = np.linspace(
            df["weekend_screen_time"].min(), df["weekend_screen_time"].max(), 100
        )
        ax.plot(
            xs, m * xs + b, color=ACCENT, linewidth=2.5, label=f"y = {m:.2f}x + {b:.2f}"
        )
        ax.set_xlabel("Weekend Screen Time (hrs)")
        ax.set_ylabel("Daily Screen Time (hrs)")
        ax.set_title(
            "Strong Linear Habit: Weekend ↔ Daily Screen Time",
            color=ACCENT,
            fontweight="bold",
            fontsize=13,
        )
        ax.legend()
        ax.grid(True)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── Gender ───────────────────────────────────────────────────────────────
    with tab3:
        st.markdown("### Gender × Addiction Level Cross-Tab")
        fig, ax = plt.subplots(figsize=(10, 5))
        ct = pd.crosstab(
            df["gender"], [df["addicted_label"], df["addiction_level"]], normalize=True
        )
        ct.plot(
            kind="bar",
            stacked=False,
            ax=ax,
            color=PALETTE[: len(ct.columns)],
            edgecolor="#0f1117",
            width=0.7,
        )
        ax.set_title(
            "Gender-wise Addiction Distribution",
            color=ACCENT,
            fontweight="bold",
            fontsize=13,
        )
        ax.tick_params(axis="x", rotation=0)
        ax.legend(fontsize=7, ncol=3)
        ax.grid(axis="y")
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown(
            """
        <div class="insight-box">
        <strong>Male Higher Addiction Rate:</strong> Males show a higher relative proportion of
        moderate/severe addiction. However, ANOVA tests (below) confirm gender differences
        in raw feature means are <em>not statistically significant</em>.
        </div>
        """,
            unsafe_allow_html=True,
        )

    # ── Correlation ──────────────────────────────────────────────────────────
    with tab4:
        st.markdown("### Correlation Heatmap")
        cat_cols = [c for c in df.columns if df[c].dtype in ["object", "category"]]
        tr_df_heat = pd.get_dummies(df, columns=cat_cols, prefix_sep="_", dtype="int")
        tr_df_heat["addicted_label"] = df["addicted_label"]

        fig, ax = plt.subplots(figsize=(14, 12))
        corr = tr_df_heat.corr(numeric_only=True)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr,
            ax=ax,
            annot=True,
            fmt=".1f",
            cmap="YlGnBu",
            mask=mask,
            linewidths=0.3,
            linecolor="#0f1117",
            annot_kws={"size": 7},
        )
        ax.set_title(
            "Feature Correlation Matrix", color=ACCENT, fontweight="bold", fontsize=14
        )
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        insights_corr = [
            "Daily & weekend screen time share R ≈ 1.0 — they encode the same habitual pattern.",
            "addiction_level_Mild is perfectly negatively correlated with addicted_label (mild ≡ not addicted).",
            "social_media_hours and app_opens_per_day show moderate positive correlation with addiction.",
            "Most categorical dummies correlate weakly with target — addiction is a complex behavioural outcome.",
        ]
        for ins in insights_corr:
            st.markdown(
                f'<div class="insight-box">• {ins}</div>', unsafe_allow_html=True
            )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — STATISTICAL TESTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Statistical Tests":
    st.markdown(
        '<div class="section-title">Statistical Tests</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="section-sub">ANOVA · Levene · OLS · NaN imputation rationale</div>',
        unsafe_allow_html=True,
    )

    test_tab1, test_tab2, test_tab3 = st.tabs(
        ["ANOVA by Age Group", "ANOVA by Gender", "NaN Imputation Proof"]
    )

    with test_tab1:
        st.markdown(
            "### One-Way ANOVA across Age Groups (18–22 / 22–27 / 27–31 / 31–35)"
        )
        anova_age = run_anova_analysis(
            df, "age_group", ["18-22", "22-27", "27-31", "31-35"]
        )
        st.dataframe(
            anova_age.style.map(
                lambda v: "color: #f0c040" if v == "✅ Yes" else "color: #888",
                subset=["Significant (α=0.05)"],
            ),
            use_container_width=True,
        )
        st.markdown(
            """
        <div class="insight-box">
        <strong>Conclusion:</strong> Despite visual patterns, most features show no statistically
        significant mean difference across age groups (p > 0.05). Age alone is <em>not</em>
        a strong predictor of usage patterns in this dataset.
        </div>
        """,
            unsafe_allow_html=True,
        )

    with test_tab2:
        st.markdown("### One-Way ANOVA across Gender Groups (Female / Male / Other)")
        anova_gen = run_anova_analysis(df, "gender", ["Female", "Male", "Other"])
        st.dataframe(
            anova_gen.style.map(
                lambda v: "color: #f0c040" if v == "✅ Yes" else "color: #888",
                subset=["Significant (α=0.05)"],
            ),
            use_container_width=True,
        )
        st.markdown(
            """
        <div class="insight-box">
        <strong>Conclusion:</strong> Gender differences in feature means are also not statistically
        significant — gender influences addiction indirectly (e.g. via behaviour patterns) rather
        than through raw usage metrics.
        </div>
        """,
            unsafe_allow_html=True,
        )

    with test_tab3:
        st.markdown("### Why NaN Addiction Level → Mild?")
        st.markdown(
            "Euclidean distance between the NaN-group's numeric profile and each known level:"
        )
        dist_df = pd.DataFrame(
            {
                "Addiction Level": list(nan_distances.keys()),
                "Squared Euclidean Distance": [
                    round(v, 2) for v in nan_distances.values()
                ],
            }
        ).sort_values("Squared Euclidean Distance")
        st.dataframe(dist_df, use_container_width=True, hide_index=True)

        fig, ax = plt.subplots(figsize=(6, 3.5))
        colors_bar = [
            PALETTE[3] if k == "Mild" else PALETTE[0] for k in nan_distances.keys()
        ]
        ax.barh(
            list(nan_distances.keys()),
            list(nan_distances.values()),
            color=colors_bar,
            edgecolor="#0f1117",
        )
        ax.set_xlabel("Squared Euclidean Distance")
        ax.set_title("NaN Profile Closest to → Mild", color=ACCENT, fontweight="bold")
        ax.grid(axis="x")
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown(
            """
        <div class="insight-box">
        <strong>Imputation Rationale:</strong> The NaN group's mean numeric profile is closest (lowest
        distance) to the <em>Mild</em> group, confirming that filling NaN → "Mild" is data-driven,
        not arbitrary.
        </div>
        """,
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL REPORT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Report":
    st.markdown('<div class="section-title">Model Report</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">gradient boosting · random forest · logistic regression · comparison</div>',
        unsafe_allow_html=True,
    )

    md = model_data

    # KPI row
    st.markdown(
        f"""
    <div class="kpi-row">
      <div class="kpi-card"><div class="kpi-value">{md['acc']*100:.1f}%</div><div class="kpi-label">GB ACCURACY</div></div>
      <div class="kpi-card"><div class="kpi-value">{md['auc']:.3f}</div><div class="kpi-label">GB ROC-AUC</div></div>
      <div class="kpi-card"><div class="kpi-value">{md['rf_acc']*100:.1f}%</div><div class="kpi-label">RF ACCURACY</div></div>
      <div class="kpi-card"><div class="kpi-value">{md['rf_auc']:.3f}</div><div class="kpi-label">RF ROC-AUC</div></div>
      <div class="kpi-card"><div class="kpi-value">{md['lg_acc']*100:.1f}%</div><div class="kpi-label">LOGIT ACCURACY</div></div>
      <div class="kpi-card"><div class="kpi-value">{md['lg_auc']:.3f}</div><div class="kpi-label">LOGIT ROC-AUC</div></div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Confusion Matrix — Gradient Boosting")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            md["cm"],
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            ax=ax,
            linewidths=0.5,
            linecolor="#0f1117",
            xticklabels=["Predicted 0", "Predicted 1"],
            yticklabels=["Actual 0", "Actual 1"],
            annot_kws={"size": 14, "weight": "bold"},
        )
        ax.set_title("Confusion Matrix", color=ACCENT, fontweight="bold")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.markdown("### Model Comparison")
        comp = pd.DataFrame(
            {
                "Model": [
                    "Logistic Regression",
                    "Random Forest",
                    "Gradient Boosting ★",
                ],
                "Accuracy": [
                    f"{md['lg_acc']*100:.2f}%",
                    f"{md['rf_acc']*100:.2f}%",
                    f"{md['acc']*100:.2f}%",
                ],
                "ROC-AUC": [
                    f"{md['lg_auc']:.4f}",
                    f"{md['rf_auc']:.4f}",
                    f"{md['auc']:.4f}",
                ],
                "Sampling": ["Borderline-SMOTE", "None", "Borderline-SMOTE"],
            }
        )
        st.dataframe(comp, use_container_width=True, hide_index=True)

        st.markdown(
            """
        <div class="insight-box">
        <strong>Why Gradient Boosting wins:</strong> Sequential error correction + Borderline-SMOTE
        (which focuses on difficult minority boundary cases) together handle the class imbalance
        and non-linear feature interactions better than the other two approaches.
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("### Classification Report — Gradient Boosting")
    report_df = pd.DataFrame(md["report"]).T.round(3)
    st.dataframe(
        report_df.style.background_gradient(
            cmap="YlGnBu", axis=None, subset=["precision", "recall", "f1-score"]
        ),
        use_container_width=True,
    )

    st.markdown("### Feature Importance")
    feat_imp = pd.Series(
        md["model"].feature_importances_, index=md["features"]
    ).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(
        feat_imp.index,
        feat_imp.values,  # type: ignore
        color=[ACCENT if v == feat_imp.max() else PALETTE[1] for v in feat_imp.values],
        edgecolor="#0f1117",
    )
    ax.set_title(
        "Feature Importances (Gradient Boosting)",
        color=ACCENT,
        fontweight="bold",
        fontsize=13,
    )
    ax.set_xlabel("Importance")
    ax.grid(axis="x")
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("### ROC Curve")
    fig, ax = plt.subplots(figsize=(6, 5))
    roc_disp = RocCurveDisplay.from_estimator(
        md["model"],
        md["x_test"],
        md["y_test"],
        ax=ax,
        name="Gradient Boosting",
    )

    roc_disp.line_.set_color(ACCENT)  # type: ignore
    roc_disp.line_.set_linewidth(2.5)  # type: ignore
    ax.plot([0, 1], [0, 1], "--", color="#555", linewidth=1)
    ax.set_title("ROC Curve", color=ACCENT, fontweight="bold")
    ax.grid(True)
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Predict Addiction":
    st.markdown(
        '<div class="section-title">Predict Addiction</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="section-sub">enter user profile · get instant addiction prediction</div>',
        unsafe_allow_html=True,
    )

    FEATURES = model_data["features"]
    model = model_data["model"]

    st.markdown("### Enter Usage Profile")
    col1, col2 = st.columns(2)

    with col1:
        daily_screen = st.slider("📱 Daily Screen Time (hours)", 0.0, 16.0, 6.0, 0.5)
        social_media = st.slider("💬 Social Media Hours/day", 0.0, 10.0, 3.0, 0.5)
        gaming = st.slider("🎮 Gaming Hours/day", 0.0, 8.0, 1.0, 0.5)
        work_study = st.slider("📚 Work/Study Hours/day", 0.0, 10.0, 4.0, 0.5)

    with col2:
        app_opens = st.number_input(
            "📲 App Opens per Day", min_value=0, max_value=500, value=80
        )
        stress_low = st.selectbox("😌 Stress Level", ["Low", "Medium", "High"], index=1)
        academic_imp = st.selectbox("📖 Academic/Work Impact", ["Yes", "No"], index=1)

    stress_low_val = 1 if stress_low == "Low" else 0
    academic_imp_val = 1 if academic_imp == "Yes" else 0

    input_data = pd.DataFrame(
        [
            [
                daily_screen,
                social_media,
                stress_low_val,
                app_opens,
                academic_imp_val,
                gaming,
                work_study,
            ]
        ],
        columns=FEATURES,
    )

    st.markdown("---")
    if st.button("🔍 Predict Now", use_container_width=True):
        logger.info(
            f"Prediction request - Input: daily_screen={daily_screen}h, social_media={social_media}h, gaming={gaming}h"
        )
        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0]
        addiction_prob = prob[1] * 100
        logger.info(
            f"Prediction result: {('ADDICTED' if pred == 1 else 'NOT ADDICTED')} (probability: {addiction_prob:.2f}%)"
        )

        if pred == 1:
            st.markdown(
                f"""
            <div class="pred-addicted">
              <div class="pred-label" style="color:#ff4444">⚠️ ADDICTED</div>
              <div class="pred-prob">Addiction Probability: <strong>{addiction_prob:.1f}%</strong></div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
            <div class="pred-not-addicted">
              <div class="pred-label" style="color:#44ff88">✅ NOT ADDICTED</div>
              <div class="pred-prob">Addiction Probability: <strong>{addiction_prob:.1f}%</strong></div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown("#### Probability Breakdown")
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.barh(
            ["Not Addicted", "Addicted"],
            [prob[0] * 100, prob[1] * 100],
            color=[PALETTE[3], PALETTE[2]],
            edgecolor="#0f1117",
            height=0.45,
        )
        for i, v in enumerate([prob[0] * 100, prob[1] * 100]):
            ax.text(v + 0.5, i, f"{v:.1f}%", va="center", color="#ccc")
        ax.set_xlim(0, 110)
        ax.set_xlabel("Probability (%)")
        ax.set_title("Model Confidence", color=ACCENT, fontweight="bold")
        ax.grid(axis="x")
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown("#### Input Summary")
        inp_display = pd.DataFrame(
            {
                "Feature": [
                    "Daily Screen Time",
                    "Social Media Hours",
                    "Gaming Hours",
                    "Work/Study Hours",
                    "App Opens/Day",
                    "Stress Level",
                    "Academic Impact",
                ],
                "Value": [
                    f"{daily_screen}h",
                    f"{social_media}h",
                    f"{gaming}h",
                    f"{work_study}h",
                    str(app_opens),
                    stress_low,
                    academic_imp,
                ],
            }
        )
        st.dataframe(inp_display, use_container_width=True, hide_index=True)

    else:
        st.markdown(
            """
        <div class="insight-box">
        Adjust the sliders and inputs above, then click <strong>Predict Now</strong>
        to get an instant addiction classification from the Gradient Boosting model.
        </div>
        """,
            unsafe_allow_html=True,
        )
