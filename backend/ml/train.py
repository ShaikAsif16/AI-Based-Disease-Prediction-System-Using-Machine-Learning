"""
Reproducible Machine Learning Training & Evaluation Pipeline
=============================================================
Project: AI-Based Disease Prediction System Using Machine Learning
Student: Shaik Asif (2403A52220) | Subject: AIDL

Workflow:
1. Load the genuine Pima Indians Diabetes dataset (no synthetic fallback).
2. Perform Exploratory Data Analysis (EDA) and record invalid-zero counts.
3. Split into stratified 80% training / 20% hold-out test sets (seed=42).
4. Build leakage-free scikit-learn Pipelines (InvalidZeroToNaN -> Median
   Imputation -> Standard Scaling -> Classifier) fitted strictly on training data.
5. Evaluate 4 algorithms (Logistic Regression, Decision Tree, Random Forest,
   SVM) using 5-fold Stratified Cross-Validation on the training set.
6. Select the best model using validation metrics prioritizing clinical
   sensitivity (Recall), discrimination (ROC-AUC), and F1-Score.
7. Evaluate all 4 trained pipelines on the untouched 20% stratified test set.
8. Persist the selected fitted pipeline (joblib), structured JSON metrics,
   model metadata, and Matplotlib/Seaborn visual reports.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path so `backend.ml.preprocessing` is the canonical
# module path when serializing custom transformers with joblib.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from backend.ml.preprocessing import (
    DIABETES_FEATURE_SPECS,
    FEATURE_ORDER,
    ZERO_AS_MISSING_COLUMNS,
    build_model_pipeline,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

DATASET_PATH = PROJECT_ROOT / "dataset" / "pima_indians_diabetes.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

PIPELINE_OUTPUT_PATH = MODELS_DIR / "diabetes_pipeline.joblib"
METADATA_OUTPUT_PATH = MODELS_DIR / "model_metadata.json"
METRICS_OUTPUT_PATH = MODELS_DIR / "evaluation_metrics.json"


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the authentic Pima Indians Diabetes CSV or raise an informative error."""
    if not path.exists():
        raise FileNotFoundError(
            f"Real dataset not found at '{path}'.\n"
            "Run `python dataset/download_dataset.py` or follow "
            "`dataset/DATASET_INSTRUCTIONS.md` to obtain the Pima Indians Diabetes dataset.\n"
            "This pipeline does not generate synthetic placeholder data."
        )

    df = pd.read_csv(path)
    expected_cols = FEATURE_ORDER + ["Outcome"]
    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    return df[expected_cols].copy()


def compute_eda_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Compute genuine exploratory data analysis statistics from the raw dataset."""
    total_rows = int(len(df))
    positive_count = int((df["Outcome"] == 1).sum())
    negative_count = int((df["Outcome"] == 0).sum())

    explicit_missing = {col: int(df[col].isna().sum()) for col in FEATURE_ORDER}
    invalid_zero_counts = {
        col: int((df[col] == 0).sum()) for col in ZERO_AS_MISSING_COLUMNS
    }
    invalid_zero_percentages = {
        col: round((count / total_rows) * 100.0, 2)
        for col, count in invalid_zero_counts.items()
    }

    # Feature summary statistics after treating physiologically invalid zeros as NaN (for EDA only)
    eda_df = df.copy()
    for col in ZERO_AS_MISSING_COLUMNS:
        eda_df[col] = eda_df[col].replace(0, np.nan)

    feature_stats: dict[str, dict[str, float | int]] = {}
    for col in FEATURE_ORDER:
        series = eda_df[col].dropna()
        feature_stats[col] = {
            "valid_count": int(series.count()),
            "invalid_zero_count": int(invalid_zero_counts.get(col, 0)),
            "mean": round(float(series.mean()), 3),
            "std": round(float(series.std()), 3),
            "min": round(float(series.min()), 3),
            "q25": round(float(series.quantile(0.25)), 3),
            "median": round(float(series.median()), 3),
            "q75": round(float(series.quantile(0.75)), 3),
            "max": round(float(series.max()), 3),
        }

    corr_with_outcome = {
        col: round(float(eda_df[col].corr(eda_df["Outcome"])), 4)
        for col in FEATURE_ORDER
    }

    return {
        "dataset_name": "Pima Indians Diabetes Database (NIDDK / UCI)",
        "total_samples": total_rows,
        "num_features": len(FEATURE_ORDER),
        "class_distribution": {
            "negative": negative_count,
            "positive": positive_count,
            "negative_percentage": round((negative_count / total_rows) * 100.0, 2),
            "positive_percentage": round((positive_count / total_rows) * 100.0, 2),
        },
        "explicit_missing_values": explicit_missing,
        "invalid_zero_counts": invalid_zero_counts,
        "invalid_zero_percentages": invalid_zero_percentages,
        "feature_statistics": feature_stats,
        "correlation_with_outcome": corr_with_outcome,
    }


def get_candidate_models() -> dict[str, dict[str, Any]]:
    """Define the 4 candidate ML models with reproducible seeds and balanced class weighting."""
    return {
        "logistic_regression": {
            "display_name": "Logistic Regression",
            "short_name": "LogReg",
            "family": "Linear Probabilistic Classifier",
            "description": (
                "Linear log-odds model with L2 regularization and balanced class weights. "
                "Offers strong calibration, interpretability, and reliable generalization on tabular clinical data."
            ),
            "estimator": LogisticRegression(
                C=1.0,
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            "scale_features": True,
        },
        "decision_tree": {
            "display_name": "Decision Tree",
            "short_name": "DecTree",
            "family": "Non-Linear Rule-Based Classifier",
            "description": (
                "Depth-regularized CART decision tree (max_depth=5, min_samples_leaf=8) "
                "with balanced class weights to capture non-linear feature thresholds while limiting overfitting."
            ),
            "estimator": DecisionTreeClassifier(
                max_depth=5,
                min_samples_leaf=8,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            "scale_features": True,
        },
        "random_forest": {
            "display_name": "Random Forest",
            "short_name": "RandForest",
            "family": "Bagged Ensemble Classifier",
            "description": (
                "Ensemble of 200 regularized decision trees (max_depth=8, min_samples_leaf=4) "
                "using bootstrap aggregation and balanced class weights to capture feature interactions."
            ),
            "estimator": RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=4,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "scale_features": True,
        },
        "svm": {
            "display_name": "Support Vector Machine (SVM)",
            "short_name": "SVM (RBF)",
            "family": "Kernel Margin Classifier",
            "description": (
                "RBF-kernel Support Vector Classifier with balanced class weights and Platt-scaled "
                "probability estimates for smooth non-linear decision boundaries."
            ),
            "estimator": CalibratedClassifierCV(
                estimator=SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
                cv=5,
                ensemble=False,
            ),
            "scale_features": True,
        },
    }


def extract_feature_importance(pipeline: Any, feature_names: list[str]) -> list[dict[str, Any]]:
    """Extract normalized relative feature importance or standardized linear magnitude."""
    clf = pipeline.named_steps["classifier"]
    raw_scores: np.ndarray | None = None
    method = "not_available"

    if hasattr(clf, "feature_importances_"):
        raw_scores = np.asarray(clf.feature_importances_, dtype=float)
        method = "gini_importance"
    elif hasattr(clf, "coef_"):
        raw_scores = np.abs(np.asarray(clf.coef_[0], dtype=float))
        method = "abs_standardized_coefficient"

    if raw_scores is None or float(raw_scores.sum()) == 0.0:
        return []

    norm = raw_scores / raw_scores.sum()
    items = []
    for idx, name in enumerate(feature_names):
        entry: dict[str, Any] = {
            "feature": name,
            "importance": round(float(norm[idx]), 4),
            "method": method,
        }
        if hasattr(clf, "coef_"):
            entry["signed_coefficient"] = round(float(clf.coef_[0][idx]), 4)
        items.append(entry)

    items.sort(key=lambda x: x["importance"], reverse=True)
    return items


def generate_visualizations(
    df: pd.DataFrame,
    eda_summary: dict[str, Any],
    model_results: dict[str, dict[str, Any]],
    selected_key: str,
) -> list[str]:
    """Generate publication-quality Matplotlib/Seaborn reports from genuine training outputs."""
    REPORTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
        }
    )

    saved_files: list[str] = []

    # 1. Class distribution & Invalid Zeros analysis
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    class_counts = [
        eda_summary["class_distribution"]["negative"],
        eda_summary["class_distribution"]["positive"],
    ]
    bars = axes[0].bar(
        ["Non-Diabetic (0)", "Diabetic (1)"],
        class_counts,
        color=["#0F2942", "#06B6D4"],
        width=0.55,
    )
    axes[0].set_title("Target Class Distribution (N = 768)")
    axes[0].set_ylabel("Number of Patients")
    for bar, count in zip(bars, class_counts):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 10,
            f"{count} ({count / 768 * 100:.1f}%)",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    zero_cols = list(eda_summary["invalid_zero_counts"].keys())
    zero_vals = [eda_summary["invalid_zero_counts"][c] for c in zero_cols]
    zbars = axes[1].barh(zero_cols, zero_vals, color="#2563EB")
    axes[1].set_title("Physiologically Invalid Zero Values Detected")
    axes[1].set_xlabel("Zero-Value Count (Treated as Missing)")
    for bar, val in zip(zbars, zero_vals):
        axes[1].text(
            val + 5,
            bar.get_y() + bar.get_height() / 2,
            f"{val} ({val / 768 * 100:.1f}%)",
            va="center",
            fontsize=9,
        )
    axes[1].set_xlim(0, max(zero_vals) + 85)
    plt.tight_layout()
    p1 = REPORTS_FIGURES_DIR / "eda_class_and_zeros.png"
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)
    saved_files.append(p1.name)

    # 2. Feature Distributions (after replacing invalid zeros with NaN for clean visualization)
    eda_df = df.copy()
    for col in ZERO_AS_MISSING_COLUMNS:
        eda_df[col] = eda_df[col].replace(0, np.nan)

    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5))
    axes_flat = axes.flatten()
    for idx, col in enumerate(FEATURE_ORDER):
        ax = axes_flat[idx]
        sns.histplot(
            data=eda_df,
            x=col,
            hue="Outcome",
            kde=True,
            ax=ax,
            palette={0: "#1E3A5F", 1: "#06B6D4"},
            alpha=0.55,
            legend=(idx == 0),
        )
        ax.set_title(col)
    fig.suptitle("Clinical Feature Distributions by Diabetes Outcome (Invalid Zeros Excluded)", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    p2 = REPORTS_FIGURES_DIR / "eda_feature_distributions.png"
    fig.savefig(p2, bbox_inches="tight")
    plt.close(fig)
    saved_files.append(p2.name)

    # 3. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    corr_matrix = eda_df.corr()
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.6,
        ax=ax,
    )
    ax.set_title("Feature Correlation Matrix (Invalid Zeros Masked as NaN)")
    plt.tight_layout()
    p3 = REPORTS_FIGURES_DIR / "eda_correlation_heatmap.png"
    fig.savefig(p3, bbox_inches="tight")
    plt.close(fig)
    saved_files.append(p3.name)

    # 4. Algorithm Performance Comparison (Hold-Out Test Set)
    metrics_to_plot = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall (Sensitivity)", "F1-Score", "ROC-AUC"]
    rows = []
    for key, res in model_results.items():
        label = res["display_name"] + (" [Selected]" if key == selected_key else "")
        for m, m_label in zip(metrics_to_plot, metric_labels):
            rows.append(
                {
                    "Algorithm": label,
                    "Metric": m_label,
                    "Score": res["test_metrics"][m],
                }
            )
    comp_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    sns.barplot(
        data=comp_df,
        x="Metric",
        y="Score",
        hue="Algorithm",
        ax=ax,
        palette=["#0F2942", "#64748B", "#2563EB", "#06B6D4"],
    )
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Hold-Out Test Set Model Comparison (Stratified 80/20 Split)")
    ax.set_ylabel("Score")
    ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=8, padding=2)
    plt.tight_layout()
    p4 = REPORTS_FIGURES_DIR / "model_comparison.png"
    fig.savefig(p4, bbox_inches="tight")
    plt.close(fig)
    saved_files.append(p4.name)

    # 5. Confusion Matrices (2x2 Grid)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.5))
    axes_flat = axes.flatten()
    for idx, (key, res) in enumerate(model_results.items()):
        cm = np.array(res["confusion_matrix"]["matrix"])
        ax = axes_flat[idx]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Pred: Non-Diabetic", "Pred: Diabetic"],
            yticklabels=["Actual: Non-Diabetic", "Actual: Diabetic"],
            ax=ax,
            annot_kws={"size": 13, "weight": "bold"},
        )
        star = " [SELECTED]" if key == selected_key else ""
        ax.set_title(
            f"{res['display_name']}{star}\n"
            f"Recall: {res['test_metrics']['recall']:.3f} | ROC-AUC: {res['test_metrics']['roc_auc']:.3f}"
        )
    fig.suptitle("Confusion Matrices on Hold-Out Test Set (N = 154)", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    p5 = REPORTS_FIGURES_DIR / "confusion_matrices.png"
    fig.savefig(p5, bbox_inches="tight")
    plt.close(fig)
    saved_files.append(p5.name)

    # 6. ROC Curves Overlay
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    colors = {
        "logistic_regression": "#0F2942",
        "decision_tree": "#94A3B8",
        "random_forest": "#2563EB",
        "svm": "#06B6D4",
    }
    for key, res in model_results.items():
        pts = res["roc_curve"]
        fpr = [p["fpr"] for p in pts]
        tpr = [p["tpr"] for p in pts]
        lw = 2.8 if key == selected_key else 1.8
        star = " [Selected]" if key == selected_key else ""
        ax.plot(
            fpr,
            tpr,
            color=colors.get(key, "#2563EB"),
            lw=lw,
            label=f"{res['display_name']}{star} (AUC = {res['test_metrics']['roc_auc']:.3f})",
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="#CBD5E1", lw=1.5, label="Random Chance (AUC = 0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)")
    ax.set_title("Receiver Operating Characteristic (ROC) Curves — Hold-Out Test Set")
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    p6 = REPORTS_FIGURES_DIR / "roc_curves.png"
    fig.savefig(p6, bbox_inches="tight")
    plt.close(fig)
    saved_files.append(p6.name)

    return saved_files


def train_and_evaluate() -> dict[str, Any]:
    """Run the full training, validation, model selection, and persistence workflow."""
    print("=" * 72)
    print("AI-Based Disease Prediction System — Real ML Training Pipeline")
    print("=" * 72)

    df = load_dataset(DATASET_PATH)
    print(f"[1/6] Loaded dataset: {DATASET_PATH.name} ({df.shape[0]} rows, {df.shape[1]} columns)")

    eda_summary = compute_eda_summary(df)
    print(
        f"[2/6] EDA complete | Positive: {eda_summary['class_distribution']['positive']} "
        f"({eda_summary['class_distribution']['positive_percentage']}%), "
        f"Negative: {eda_summary['class_distribution']['negative']} "
        f"({eda_summary['class_distribution']['negative_percentage']}%)"
    )
    print(f"      Invalid zeros detected: {eda_summary['invalid_zero_counts']}")

    X = df[FEATURE_ORDER].copy()
    y = df["Outcome"].astype(int).copy()

    # Stratified split BEFORE any imputation or scaling to guarantee zero data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    print(
        f"[3/6] Stratified split complete | Train: {len(X_train)} "
        f"(pos={int(y_train.sum())}), Test: {len(X_test)} (pos={int(y_test.sum())})"
    )

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    candidates = get_candidate_models()
    fitted_pipelines: dict[str, Any] = {}
    model_results: dict[str, dict[str, Any]] = {}

    print(f"[4/6] Training & evaluating {len(candidates)} models ({CV_FOLDS}-fold Stratified CV + Hold-Out Test)...")
    for key, spec in candidates.items():
        pipeline = build_model_pipeline(
            estimator=spec["estimator"],
            scale_features=spec["scale_features"],
        )

        # Cross-validation strictly inside training set (each fold fits its own imputer + scaler)
        cv_scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
        )

        cv_metrics = {
            "accuracy": round(float(np.mean(cv_scores["test_accuracy"])), 4),
            "accuracy_std": round(float(np.std(cv_scores["test_accuracy"])), 4),
            "precision": round(float(np.mean(cv_scores["test_precision"])), 4),
            "precision_std": round(float(np.std(cv_scores["test_precision"])), 4),
            "recall": round(float(np.mean(cv_scores["test_recall"])), 4),
            "recall_std": round(float(np.std(cv_scores["test_recall"])), 4),
            "f1_score": round(float(np.mean(cv_scores["test_f1"])), 4),
            "f1_score_std": round(float(np.std(cv_scores["test_f1"])), 4),
            "roc_auc": round(float(np.mean(cv_scores["test_roc_auc"])), 4),
            "roc_auc_std": round(float(np.std(cv_scores["test_roc_auc"])), 4),
        }

        # Clinical screening validation composite score (prioritizing Sensitivity / Recall + ROC-AUC + F1)
        screening_selection_score = round(
            0.45 * cv_metrics["recall"]
            + 0.35 * cv_metrics["roc_auc"]
            + 0.20 * cv_metrics["f1_score"],
            4,
        )
        cv_metrics["clinical_selection_score"] = screening_selection_score

        # Fit pipeline on full training split (imputer & scaler fitted ONLY on X_train)
        pipeline.fit(X_train, y_train)
        fitted_pipelines[key] = pipeline

        # Evaluate on untouched hold-out test set
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        test_metrics = {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "specificity": round(float(specificity), 4),
            "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        }

        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_points = [
            {
                "fpr": round(float(f), 4),
                "tpr": round(float(t), 4),
                "threshold": round(float(th), 4) if np.isfinite(th) else None,
            }
            for f, t, th in zip(fpr, tpr, thresholds)
        ]

        feature_importance = extract_feature_importance(pipeline, FEATURE_ORDER)

        model_results[key] = {
            "key": key,
            "display_name": spec["display_name"],
            "short_name": spec["short_name"],
            "family": spec["family"],
            "description": spec["description"],
            "cv_metrics": cv_metrics,
            "test_metrics": test_metrics,
            "confusion_matrix": {
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "matrix": [[tn, fp], [fn, tp]],
            },
            "roc_curve": roc_points,
            "feature_importance": feature_importance,
            "supports_probability": hasattr(pipeline, "predict_proba"),
        }

        print(
            f"      - {spec['display_name']:<28} | "
            f"CV Recall: {cv_metrics['recall']:.4f}, CV AUC: {cv_metrics['roc_auc']:.4f}, "
            f"Score: {screening_selection_score:.4f} || "
            f"Test Acc: {test_metrics['accuracy']:.4f}, Rec: {test_metrics['recall']:.4f}, "
            f"F1: {test_metrics['f1_score']:.4f}, AUC: {test_metrics['roc_auc']:.4f}"
        )

    # Select best model strictly from validation metrics (no test-set peeking)
    selected_key = max(
        model_results.keys(),
        key=lambda k: (
            model_results[k]["cv_metrics"]["clinical_selection_score"],
            model_results[k]["cv_metrics"]["roc_auc"],
        ),
    )
    selected_result = model_results[selected_key]
    selected_pipeline = fitted_pipelines[selected_key]

    print(
        f"\n[5/6] Selected Model (via 5-fold CV Clinical Screening Score): "
        f"{selected_result['display_name']}"
    )
    print(
        f"      CV Clinical Score = {selected_result['cv_metrics']['clinical_selection_score']:.4f} | "
        f"Test Recall = {selected_result['test_metrics']['recall']:.4f} | "
        f"Test ROC-AUC = {selected_result['test_metrics']['roc_auc']:.4f}"
    )

    # Extract fitted training-only imputation medians for transparency
    fitted_imputer = selected_pipeline.named_steps["median_imputer"]
    training_medians = {
        col: round(float(val), 4)
        for col, val in zip(FEATURE_ORDER, fitted_imputer.statistics_)
    }

    # Save artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected_pipeline, PIPELINE_OUTPUT_PATH)

    generated_figures = generate_visualizations(
        df=df,
        eda_summary=eda_summary,
        model_results=model_results,
        selected_key=selected_key,
    )

    trained_at = datetime.now(timezone.utc).isoformat()
    selection_rationale = (
        f"{selected_result['display_name']} achieved the highest 5-fold stratified cross-validation "
        f"clinical screening score ({selected_result['cv_metrics']['clinical_selection_score']:.4f}), "
        f"defined as 0.45×Recall + 0.35×ROC-AUC + 0.20×F1-Score on the training set. "
        "Prioritizing sensitivity (Recall) alongside ROC-AUC minimizes missed high-risk cases "
        "(False Negatives) while preserving reliable overall discrimination without peeking at the hold-out test set."
    )

    model_metadata = {
        "disease_id": "diabetes",
        "disease_name": "Type 2 Diabetes Risk Screening",
        "selected_model_key": selected_key,
        "selected_model_name": selected_result["display_name"],
        "model_family": selected_result["family"],
        "model_description": selected_result["description"],
        "supports_probability": selected_result["supports_probability"],
        "selection_criteria": {
            "method": "5-fold Stratified Cross-Validation on Training Set (80%)",
            "formula": "0.45 * CV_Recall + 0.35 * CV_ROC_AUC + 0.20 * CV_F1",
            "rationale": selection_rationale,
        },
        "training_configuration": {
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "cv_folds": CV_FOLDS,
            "stratified": True,
            "leakage_prevention": (
                "Train/test split performed before preprocessing. Invalid zero replacement, "
                "median imputation, and z-score standardization are encapsulated inside a "
                "scikit-learn Pipeline fitted exclusively on training folds."
            ),
            "class_weighting": "balanced",
            "training_fitted_medians": training_medians,
        },
        "features": [spec.to_dict() for spec in DIABETES_FEATURE_SPECS],
        "feature_order": FEATURE_ORDER,
        "zero_as_missing_columns": ZERO_AS_MISSING_COLUMNS,
        "artifact_path": str(PIPELINE_OUTPUT_PATH.relative_to(PROJECT_ROOT)),
        "trained_at_utc": trained_at,
        "versions": {
            "python": sys.version.split()[0],
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
        "supported_diseases": [
            {
                "id": "diabetes",
                "name": "Diabetes Risk Screening",
                "status": "active",
                "dataset": "Pima Indians Diabetes Database (NIDDK / UCI)",
            },
            {
                "id": "cardiovascular",
                "name": "Cardiovascular Disease Risk",
                "status": "planned",
                "dataset": "Extensible Registry Architecture (UCI Heart Disease Ready)",
            },
            {
                "id": "chronic_kidney",
                "name": "Chronic Kidney Disease Risk",
                "status": "planned",
                "dataset": "Extensible Registry Architecture (UCI CKD Ready)",
            },
        ],
    }

    evaluation_payload = {
        "trained_at_utc": trained_at,
        "selected_model_key": selected_key,
        "selected_model_name": selected_result["display_name"],
        "selection_rationale": selection_rationale,
        "split_summary": {
            "total_samples": int(len(df)),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "train_positive": int(y_train.sum()),
            "train_negative": int((y_train == 0).sum()),
            "test_positive": int(y_test.sum()),
            "test_negative": int((y_test == 0).sum()),
            "random_state": RANDOM_STATE,
            "cv_folds": CV_FOLDS,
        },
        "dataset_summary": eda_summary,
        "models": model_results,
        "generated_figures": generated_figures,
    }

    METADATA_OUTPUT_PATH.write_text(json.dumps(model_metadata, indent=2), encoding="utf-8")
    METRICS_OUTPUT_PATH.write_text(json.dumps(evaluation_payload, indent=2), encoding="utf-8")

    print(f"[6/6] Artifacts saved:")
    print(f"      - Pipeline : {PIPELINE_OUTPUT_PATH}")
    print(f"      - Metadata : {METADATA_OUTPUT_PATH}")
    print(f"      - Metrics  : {METRICS_OUTPUT_PATH}")
    print(f"      - Figures  : {len(generated_figures)} PNG plots in {REPORTS_FIGURES_DIR}")
    print("=" * 72)

    return evaluation_payload


if __name__ == "__main__":
    train_and_evaluate()
