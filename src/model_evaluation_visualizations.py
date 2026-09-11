import pandas as pd
import matplotlib.pyplot as plt
import os

# ============================================================
# MARKETLENS - MODEL EVALUATION VISUALIZATIONS
# ============================================================

print("=" * 70)
print("MARKETLENS - MODEL EVALUATION VISUALIZATIONS")
print("=" * 70)

# ------------------------------------------------------------
# Create output folder
# ------------------------------------------------------------

output_folder = "data/model_visualizations"
os.makedirs(output_folder, exist_ok=True)


# ------------------------------------------------------------
# 1. Classification Accuracy Comparison
# ------------------------------------------------------------

classification_models = [
    "Random Forest",
    "Gradient Boosting",
    "Logistic Regression"
]

accuracy = [48.77, 49.23, 49.23]
baseline_accuracy = 48.03

plt.figure(figsize=(9, 6))

bars = plt.bar(
    classification_models,
    accuracy
)

plt.axhline(
    y=baseline_accuracy,
    linestyle="--",
    label=f"Baseline ({baseline_accuracy:.2f}%)"
)

plt.title("Classification Model Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 60)
plt.legend()

for bar, value in zip(bars, accuracy):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.5,
        f"{value:.2f}%",
        ha="center"
    )

plt.tight_layout()

file_path = os.path.join(
    output_folder,
    "classification_accuracy_comparison.png"
)

plt.savefig(file_path, dpi=300)
plt.close()

print(f"Created: {file_path}")


# ------------------------------------------------------------
# 2. Regression MAE Comparison
# ------------------------------------------------------------

regression_models = [
    "RF Price",
    "RF Return",
    "Improved RF Return"
]

mae_values = [
    36.7940,
    1.0998,
    1.0788
]

baseline_mae = [
    21.3812,
    1.0647,
    1.0566
]

plt.figure(figsize=(10, 6))

x = range(len(regression_models))

width = 0.35

plt.bar(
    [i - width / 2 for i in x],
    mae_values,
    width,
    label="Model MAE"
)

plt.bar(
    [i + width / 2 for i in x],
    baseline_mae,
    width,
    label="Baseline MAE"
)

plt.xticks(x, regression_models)
plt.title("Regression Model MAE vs Baseline")
plt.xlabel("Model")
plt.ylabel("MAE")
plt.legend()

plt.tight_layout()

file_path = os.path.join(
    output_folder,
    "regression_mae_comparison.png"
)

plt.savefig(file_path, dpi=300)
plt.close()

print(f"Created: {file_path}")


# ------------------------------------------------------------
# 3. ROC-AUC Comparison
# ------------------------------------------------------------

roc_auc = [
    0.4900,
    0.4922,
    0.4997
]

plt.figure(figsize=(9, 6))

bars = plt.bar(
    classification_models,
    roc_auc
)

plt.axhline(
    y=0.5,
    linestyle="--",
    label="Random Chance (0.50)"
)

plt.title("ROC-AUC Comparison")
plt.xlabel("Model")
plt.ylabel("ROC-AUC")
plt.ylim(0.45, 0.55)
plt.legend()

for bar, value in zip(bars, roc_auc):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.001,
        f"{value:.4f}",
        ha="center"
    )

plt.tight_layout()

file_path = os.path.join(
    output_folder,
    "roc_auc_comparison.png"
)

plt.savefig(file_path, dpi=300)
plt.close()

print(f"Created: {file_path}")


# ------------------------------------------------------------
# 4. F1 Score Comparison
# ------------------------------------------------------------

f1_scores = [
    46.73,
    55.49,
    55.09
]

plt.figure(figsize=(9, 6))

bars = plt.bar(
    classification_models,
    f1_scores
)

plt.title("Classification F1 Score Comparison")
plt.xlabel("Model")
plt.ylabel("F1 Score (%)")
plt.ylim(0, 70)

for bar, value in zip(bars, f1_scores):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1,
        f"{value:.2f}%",
        ha="center"
    )

plt.tight_layout()

file_path = os.path.join(
    output_folder,
    "classification_f1_comparison.png"
)

plt.savefig(file_path, dpi=300)
plt.close()

print(f"Created: {file_path}")


# ------------------------------------------------------------
# 5. Classification Performance Overview
# ------------------------------------------------------------

metrics = ["Accuracy", "Precision", "Recall", "F1"]

random_forest = [48.77, 46.68, 46.79, 46.73]
gradient_boosting = [49.23, 47.93, 65.88, 55.49]
logistic_regression = [49.23, 47.90, 64.82, 55.09]

plt.figure(figsize=(11, 7))

x = range(len(metrics))
width = 0.25

plt.bar(
    [i - width for i in x],
    random_forest,
    width,
    label="Random Forest"
)

plt.bar(
    x,
    gradient_boosting,
    width,
    label="Gradient Boosting"
)

plt.bar(
    [i + width for i in x],
    logistic_regression,
    width,
    label="Logistic Regression"
)

plt.xticks(x, metrics)
plt.ylabel("Score (%)")
plt.xlabel("Evaluation Metric")
plt.title("Classification Model Performance Overview")
plt.ylim(0, 100)
plt.legend()

plt.tight_layout()

file_path = os.path.join(
    output_folder,
    "classification_performance_overview.png"
)

plt.savefig(file_path, dpi=300)
plt.close()

print(f"Created: {file_path}")


# ------------------------------------------------------------
# 6. Model Improvement Over Baseline
# ------------------------------------------------------------

model_names = [
    "Random Forest",
    "Gradient Boosting",
    "Logistic Regression"
]

improvement = [
    48.77 - 48.03,
    49.23 - 48.03,
    49.23 - 48.03
]

plt.figure(figsize=(9, 6))

bars = plt.bar(
    model_names,
    improvement
)

plt.axhline(
    y=0,
    linestyle="-"
)

plt.title("Classification Accuracy Improvement Over Baseline")
plt.xlabel("Model")
plt.ylabel("Improvement (Percentage Points)")

for bar, value in zip(bars, improvement):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.03,
        f"+{value:.2f}",
        ha="center"
    )

plt.tight_layout()

file_path = os.path.join(
    output_folder,
    "model_improvement_over_baseline.png"
)

plt.savefig(file_path, dpi=300)
plt.close()

print(f"Created: {file_path}")


# ------------------------------------------------------------
# Final message
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("VISUALIZATION GENERATION COMPLETE")
print("=" * 70)

print("\nAll visualization files are available in:")
print(output_folder)

print("\nFiles created:")

files = os.listdir(output_folder)

for file in files:
    print(f" - {file}")

print("\nThese charts can be used in:")
print(" - MarketLens dashboard")
print(" - Project report")
print(" - Project presentation")
print(" - Model evaluation section")