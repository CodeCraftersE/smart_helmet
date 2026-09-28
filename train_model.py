#!/usr/bin/env python3
"""
train_model.py
--------------
Machine Learning Training & Comparative Evaluation Pipeline for Smart Helmet System.

Models Trained:
1. Random Forest Model A (Motion Only: 9 IMU features)
2. Random Forest Model B (Motion + SPS30 Particulate Matter: 15 multimodal features)
3. Logistic Regression Model A (Motion Only with StandardScaler)
4. Logistic Regression Model B (Motion + SPS30 with StandardScaler)

Experimental Hypothesis:
"Can environmental particulate-matter exposure information improve multimodal rider fatigue-state classification?"

Saves:
- models/smart_helmet_random_forest.joblib (Final Model B)
- models/feature_names.json (Feature list for Model B)
- models/scaler.joblib (StandardScaler for Logistic Regression)
- results/confusion_matrix_motion_only.png
- results/confusion_matrix_motion_sps30.png
- results/model_comparison.png
- results/feature_importance.png
"""

import os
import json
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Constants & Paths
DATA_PATH = os.path.join("data", "smart_helmet_synthetic_10000.csv")
MODELS_DIR = "models"
RESULTS_DIR = "results"
SEED = 42

# Feature Definitions
MOTION_FEATURES = [
    "ax", "ay", "az",
    "gx", "gy", "gz",
    "tilt_angle", "acc_magnitude", "gyro_magnitude"
]

SPS30_FEATURES = [
    "pm1", "pm25", "pm4", "pm10",
    "pm25_change", "pm10_change"
]

ALL_FEATURES = MOTION_FEATURES + SPS30_FEATURES
LABEL_NAMES = ["ALERT", "DROWSY", "FATIGUED"]


def setup_directories():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


def load_and_split_data(data_path: str):
    """
    Loads dataset and performs a session-grouped stratified split:
    - 70% Training (28 sessions)
    - 15% Validation (6 sessions)
    - 15% Testing (6 sessions)
    Prevents temporal and subject leakage across time steps.
    """
    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows across {df['session_id'].nunique()} sessions and {df['subject_id'].nunique()} subjects.")

    # Group sessions by dominant class for stratified splitting
    session_summary = df.groupby("session_id")["label"].value_counts().unstack(fill_value=0)
    dominant_class = session_summary.values.argmax(axis=1)
    sessions = session_summary.index.values

    # 70% Train (28 sessions), 30% Test+Val (12 sessions)
    train_sess, test_val_sess, _, test_val_dom = train_test_split(
        sessions, dominant_class, test_size=12, stratify=dominant_class, random_state=SEED
    )

    # 15% Val (6 sessions), 15% Test (6 sessions)
    val_sess, test_sess = train_test_split(
        test_val_sess, test_size=6, stratify=test_val_dom, random_state=SEED
    )

    train_df = df[df["session_id"].isin(train_sess)].copy()
    val_df = df[df["session_id"].isin(val_sess)].copy()
    test_df = df[df["session_id"].isin(test_sess)].copy()

    print("\n--- Data Splitting Strategy: Session-Grouped Stratified Split ---")
    print(f"Train Set: {len(train_df):5d} rows ({len(train_sess)} sessions: {train_sess[:4]}...)")
    print(f"Val Set  : {len(val_df):5d} rows ({len(val_sess)} sessions: {val_sess})")
    print(f"Test Set : {len(test_df):5d} rows ({len(test_sess)} sessions: {test_sess})")

    print("\nClass distributions:")
    print("Train:", train_df["label_name"].value_counts().to_dict())
    print("Val  :", val_df["label_name"].value_counts().to_dict())
    print("Test :", test_df["label_name"].value_counts().to_dict())

    return train_df, val_df, test_df


def evaluate_model(name: str, y_true, y_pred, y_prob=None):
    """Computes comprehensive classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro")
    prec_weighted = precision_score(y_true, y_pred, average="weighted")
    rec_macro = recall_score(y_true, y_pred, average="macro")
    rec_weighted = recall_score(y_true, y_pred, average="weighted")
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "model_name": name,
        "accuracy": acc,
        "precision_macro": prec_macro,
        "precision_weighted": prec_weighted,
        "recall_macro": rec_macro,
        "recall_weighted": rec_weighted,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm
    }
    return metrics


def plot_confusion_matrix(cm, model_name: str, save_path: str):
    """Plots and saves high-resolution confusion matrix with counts and percentages."""
    plt.figure(figsize=(7, 6))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Combined label strings: Count and Row %
    annot_matrix = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot_matrix[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)"

    sns.heatmap(
        cm,
        annot=annot_matrix,
        fmt="",
        cmap="Blues",
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
        cbar=True,
        square=True,
        linewidths=1.0,
        annot_kws={"fontsize": 11, "fontweight": "bold"}
    )
    plt.title(f"Confusion Matrix: {model_name}", fontsize=14, pad=12, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=12, labelpad=8)
    plt.ylabel("True Label", fontsize=12, labelpad=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot: {save_path}")


def plot_model_comparison(metrics_a, metrics_b, save_path: str):
    """Plots comparative bar chart between Model A (Motion Only) and Model B (Motion + SPS30)."""
    metric_keys = [
        ("accuracy", "Accuracy"),
        ("precision_macro", "Macro Precision"),
        ("recall_macro", "Macro Recall"),
        ("f1_macro", "Macro F1"),
        ("f1_weighted", "Weighted F1")
    ]

    labels = [m[1] for m in metric_keys]
    vals_a = [metrics_a[m[0]] for m in metric_keys]
    vals_b = [metrics_b[m[0]] for m in metric_keys]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(10, 6))
    rects1 = plt.bar(x - width/2, vals_a, width, label="Model A (Motion Only)", color="#3498db", edgecolor="#2980b9", alpha=0.9)
    rects2 = plt.bar(x + width/2, vals_b, width, label="Model B (Motion + SPS30)", color="#2ecc71", edgecolor="#27ae60", alpha=0.9)

    plt.ylabel("Score", fontsize=12, fontweight="bold")
    plt.title("Model Comparison: Motion Only vs. Motion + SPS30 Multimodal Context", fontsize=13, fontweight="bold", pad=14)
    plt.xticks(x, labels, fontsize=11)
    plt.ylim(0.0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.legend(fontsize=11, loc="lower right")

    # Value labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            plt.annotate(f"{height:.3f}",
                         xy=(rect.get_x() + rect.get_width() / 2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved model comparison plot: {save_path}")


def plot_feature_importance(model, feature_names, save_path: str):
    """Plots and saves feature importances for Random Forest Model B."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    # Color code: blue for IMU, orange for SPS30
    colors = ["#e67e22" if f in SPS30_FEATURES else "#2980b9" for f in sorted_features]

    plt.figure(figsize=(10, 7))
    bars = plt.barh(range(len(sorted_features)), sorted_importances[::-1], color=colors[::-1], edgecolor="black", alpha=0.85)
    plt.yticks(range(len(sorted_features)), sorted_features[::-1], fontsize=11)
    plt.xlabel("Random Forest Gini Feature Importance", fontsize=12, fontweight="bold")
    plt.title("Feature Importance: Model B (Motion + SPS30 Environmental Features)", fontsize=13, fontweight="bold", pad=12)

    # Custom legend for sensor types
    import matplotlib.patches as mpatches
    imu_patch = mpatches.Patch(color='#2980b9', label='MPU6050 Motion Features')
    sps_patch = mpatches.Patch(color='#e67e22', label='SPS30 Environmental Features')
    plt.legend(handles=[imu_patch, sps_patch], loc="lower right", fontsize=11)

    plt.grid(axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved feature importance plot: {save_path}")

    # Print Top 10
    print("\n" + "=" * 60)
    print("TOP 10 FEATURES BY IMPORTANCE (Random Forest Model B)")
    print("=" * 60)
    for rank, (feat, imp) in enumerate(zip(sorted_features[:10], sorted_importances[:10]), 1):
        sensor_type = "SPS30 (Environmental)" if feat in SPS30_FEATURES else "MPU6050 (Motion)"
        print(f"  {rank:2d}. {feat:16s}: {imp:6.4f}  [{sensor_type}]")
    print("=" * 60)


def main():
    setup_directories()

    # 1. Load data and split
    train_df, val_df, test_df = load_and_split_data(DATA_PATH)

    # Prepare feature sets
    X_train_a = train_df[MOTION_FEATURES]
    X_val_a = val_df[MOTION_FEATURES]
    X_test_a = test_df[MOTION_FEATURES]

    X_train_b = train_df[ALL_FEATURES]
    X_val_b = val_df[ALL_FEATURES]
    X_test_b = test_df[ALL_FEATURES]

    y_train = train_df["label"].values
    y_val = val_df["label"].values
    y_test = test_df["label"].values

    # Preprocessing: StandardScaler for Logistic Regression
    scaler_a = StandardScaler().fit(X_train_a)
    scaler_b = StandardScaler().fit(X_train_b)

    X_train_a_scaled = scaler_a.transform(X_train_a)
    X_test_a_scaled = scaler_a.transform(X_test_a)

    X_train_b_scaled = scaler_b.transform(X_train_b)
    X_test_b_scaled = scaler_b.transform(X_test_b)

    # -------------------------------------------------------------
    # 2. Train Models
    # -------------------------------------------------------------
    print("\nTraining models...")

    # Logistic Regression (Baseline)
    print("  -> Training Logistic Regression Model A (Motion Only)...")
    lr_a = LogisticRegression(max_iter=1000, random_state=SEED)
    lr_a.fit(X_train_a_scaled, y_train)

    print("  -> Training Logistic Regression Model B (Motion + SPS30)...")
    lr_b = LogisticRegression(max_iter=1000, random_state=SEED)
    lr_b.fit(X_train_b_scaled, y_train)

    # Random Forest (Main Models)
    print("  -> Training Random Forest Model A (Motion Only)...")
    rf_a = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=4,
        random_state=SEED,
        n_jobs=-1
    )
    rf_a.fit(X_train_a, y_train)

    print("  -> Training Random Forest Model B (Motion + SPS30)...")
    rf_b = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=4,
        random_state=SEED,
        n_jobs=-1
    )
    rf_b.fit(X_train_b, y_train)

    # -------------------------------------------------------------
    # 3. Model Evaluations on Test Set
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("MODEL EVALUATION ON INDEPENDENT TEST SESSIONS (1,500 rows)")
    print("=" * 60)

    # Logistic Regression evaluations
    pred_lr_a = lr_a.predict(X_test_a_scaled)
    metrics_lr_a = evaluate_model("Logistic Regression (Motion Only)", y_test, pred_lr_a)

    pred_lr_b = lr_b.predict(X_test_b_scaled)
    metrics_lr_b = evaluate_model("Logistic Regression (Motion + SPS30)", y_test, pred_lr_b)

    # Random Forest evaluations
    pred_rf_a = rf_a.predict(X_test_a)
    metrics_rf_a = evaluate_model("Random Forest Model A (Motion Only)", y_test, pred_rf_a)

    pred_rf_b = rf_b.predict(X_test_b)
    metrics_rf_b = evaluate_model("Random Forest Model B (Motion + SPS30)", y_test, pred_rf_b)

    # Print Classification Reports
    print("\n--- Model A: Random Forest (Motion Only) ---")
    print(classification_report(y_test, pred_rf_a, target_names=LABEL_NAMES, digits=4))

    print("\n--- Model B: Random Forest (Motion + SPS30 Multimodal) ---")
    print(classification_report(y_test, pred_rf_b, target_names=LABEL_NAMES, digits=4))

    # Print Comparative Summary Table
    print("\n" + "=" * 80)
    print(f"{'Model':<35} | {'Accuracy':<9} | {'Macro F1':<9} | {'Weighted F1':<11} | {'Macro Prec':<10}")
    print("-" * 80)
    for m in [metrics_lr_a, metrics_lr_b, metrics_rf_a, metrics_rf_b]:
        print(f"{m['model_name']:<35} | {m['accuracy']:<9.4f} | {m['f1_macro']:<9.4f} | {m['f1_weighted']:<11.4f} | {m['precision_macro']:<10.4f}")
    print("=" * 80)

    # Show Difference between Model A and Model B
    delta_acc = metrics_rf_b["accuracy"] - metrics_rf_a["accuracy"]
    delta_macro_f1 = metrics_rf_b["f1_macro"] - metrics_rf_a["f1_macro"]
    delta_weighted_f1 = metrics_rf_b["f1_weighted"] - metrics_rf_a["f1_weighted"]

    print("\n--- Difference: Model B (Motion + SPS30) minus Model A (Motion Only) ---")
    print(f"  Delta Accuracy   : {delta_acc:+7.4f} ({delta_acc*100:+.2f}%)")
    print(f"  Delta Macro F1   : {delta_macro_f1:+7.4f} ({delta_macro_f1*100:+.2f}%)")
    print(f"  Delta Weighted F1: {delta_weighted_f1:+7.4f} ({delta_weighted_f1*100:+.2f}%)")

    # -------------------------------------------------------------
    # 4. Save Plots
    # -------------------------------------------------------------
    print("\nGenerating visualization artifacts...")
    plot_confusion_matrix(
        metrics_rf_a["confusion_matrix"],
        "Random Forest Model A (Motion Only)",
        os.path.join(RESULTS_DIR, "confusion_matrix_motion_only.png")
    )

    plot_confusion_matrix(
        metrics_rf_b["confusion_matrix"],
        "Random Forest Model B (Motion + SPS30)",
        os.path.join(RESULTS_DIR, "confusion_matrix_motion_sps30.png")
    )

    plot_model_comparison(
        metrics_rf_a,
        metrics_rf_b,
        os.path.join(RESULTS_DIR, "model_comparison.png")
    )

    plot_feature_importance(
        rf_b,
        ALL_FEATURES,
        os.path.join(RESULTS_DIR, "feature_importance.png")
    )

    # -------------------------------------------------------------
    # 5. Save Models and Metadata
    # -------------------------------------------------------------
    print("\nSaving trained models and artifacts...")
    rf_model_path = os.path.join(MODELS_DIR, "smart_helmet_random_forest.joblib")
    scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
    features_path = os.path.join(MODELS_DIR, "feature_names.json")

    joblib.dump(rf_b, rf_model_path)
    print(f"  Saved Random Forest model: {rf_model_path}")

    joblib.dump(scaler_b, scaler_path)
    print(f"  Saved Scaler: {scaler_path}")

    with open(features_path, "w") as f:
        json.dump(ALL_FEATURES, f, indent=2)
    print(f"  Saved feature names list: {features_path}")

    print("\nTraining and evaluation pipeline completed successfully!")


if __name__ == "__main__":
    main()
