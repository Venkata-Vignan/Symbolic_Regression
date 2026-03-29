import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pysr_runtime import configure_pysr_environment


configure_pysr_environment()

import pickle


RESULTS_DIR = PROJECT_ROOT / "Results"
RESULTS_DIR.mkdir(exist_ok=True)
BUNDLE_PATH = PROJECT_ROOT / "sr_bundle_energy_heating.pkl"

MODEL_ORDER = [
    "Linear Regression",
    "Ridge Regression",
    "Lasso Regression",
    "Random Forest",
    "Symbolic Regression",
]
MODEL_COLORS = {
    "Linear Regression": "#6c8ebf",
    "Ridge Regression": "#9a7cff",
    "Lasso Regression": "#f4a261",
    "Random Forest": "#2ec4b6",
    "Symbolic Regression": "#ff6b6b",
}


def load_bundle():
    with BUNDLE_PATH.open("rb") as file:
        return pickle.load(file)


def apply_style():
    plt.style.use("dark_background")


def save_r2_comparison(metrics):
    models = [name for name in MODEL_ORDER if name in metrics]
    values = [metrics[name]["R2"] for name in models]
    colors = [MODEL_COLORS[name] for name in models]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(models, values, color=colors)
    ax.set_title("R2 Comparison Across Models")
    ax.set_ylabel("R2")
    ax.tick_params(axis="x", rotation=20)
    for index, value in enumerate(values):
        ax.text(index, value + 0.005, f"{value:.4f}", ha="center")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "r^2_comparison.png", dpi=200)
    plt.close(fig)


def save_error_comparison(metrics):
    models = [name for name in MODEL_ORDER if name in metrics]
    rmse = [metrics[name]["RMSE"] for name in models]
    mae = [metrics[name]["MAE"] for name in models]
    x = np.arange(len(models))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - width / 2, rmse, width, label="RMSE", color="#2ec4b6")
    ax.bar(x + width / 2, mae, width, label="MAE", color="#ff6b6b")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20)
    ax.set_title("Error Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "error_comparison.png", dpi=200)
    plt.close(fig)


def save_actual_vs_predicted(evaluation_df):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        evaluation_df["Actual"],
        evaluation_df["Symbolic Regression"],
        alpha=0.8,
        color=MODEL_COLORS["Symbolic Regression"],
        edgecolors="white",
        linewidths=0.5,
    )
    min_axis = min(evaluation_df["Actual"].min(), evaluation_df["Symbolic Regression"].min())
    max_axis = max(evaluation_df["Actual"].max(), evaluation_df["Symbolic Regression"].max())
    ax.plot([min_axis, max_axis], [min_axis, max_axis], "--", color="white", linewidth=1.5)
    ax.set_title("Actual vs Predicted Heating Load (Symbolic Regression)")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "Act_pred.png", dpi=200)
    plt.close(fig)


def save_feature_importance(bundle):
    rf_model = bundle["models"]["Random Forest"]
    feature_names = bundle["feature_names"]
    importances = rf_model.feature_importances_
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(feature_names, importances, color="#2ec4b6")
    ax.set_title("Feature Importance (Random Forest)")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "Feature Importance (RF).png", dpi=200)
    plt.close(fig)


def save_symbolic_surfaces(bundle):
    feature_names = bundle["feature_names"]
    defaults = bundle["default_input"]
    ranges = bundle["valid_feature_values"]
    symbolic_model = bundle["models"]["Symbolic Regression"]
    scaler = bundle["scaler"]

    roof_vals = np.array(ranges["X4"], dtype=float)
    orientation_vals = np.array(ranges["X6"], dtype=float)
    roof_grid, orientation_grid = np.meshgrid(roof_vals, orientation_vals)

    rows = []
    for orientation in orientation_vals:
        for roof in roof_vals:
            row = defaults.copy()
            row["X4"] = float(roof)
            row["X6"] = float(orientation)
            rows.append(row)
    grid_df = pd.DataFrame(rows)[feature_names]
    preds = symbolic_model.predict(scaler.transform(grid_df)).reshape(orientation_grid.shape)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(roof_grid, orientation_grid, preds, cmap="viridis", edgecolor="none", alpha=0.95)
    ax.set_title("3D Symbolic Regression Visualization")
    ax.set_xlabel("Roof Area (X4)")
    ax.set_ylabel("Orientation (X6)")
    ax.set_zlabel("Heating Load")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "3d_visualization.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(roof_grid, orientation_grid, preds, levels=20, cmap="viridis")
    fig.colorbar(contour, ax=ax)
    ax.set_title("Heating Load Contour Map")
    ax.set_xlabel("Roof Area (X4)")
    ax.set_ylabel("Orientation (X6)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "Heating Load Contour Map.png", dpi=200)
    plt.close(fig)

    roof_curve = np.linspace(float(roof_vals.min()), float(roof_vals.max()), 300)
    rows = []
    for roof in roof_curve:
        row = defaults.copy()
        row["X4"] = float(roof)
        rows.append(row)
    curve_df = pd.DataFrame(rows)[feature_names]
    outputs = symbolic_model.predict(scaler.transform(curve_df))
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(roof_curve, outputs, color=MODEL_COLORS["Symbolic Regression"], linewidth=3)
    ax.set_title("Optimization Curve")
    ax.set_xlabel("Roof Area (X4)")
    ax.set_ylabel("Heating Load")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "Optimization Curve.png", dpi=200)
    plt.close(fig)


def save_complexity_loss(bundle):
    symbolic_model = bundle["models"]["Symbolic Regression"]
    equations = symbolic_model.equations_.copy()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(equations["complexity"], equations["loss"], marker="o", color=MODEL_COLORS["Symbolic Regression"])
    ax.set_title("Complexity vs Loss Curve")
    ax.set_xlabel("Complexity")
    ax.set_ylabel("Loss")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "complexity_vs_loss_curve.png", dpi=200)
    plt.close(fig)


def main():
    apply_style()
    bundle = load_bundle()
    metrics = bundle["metrics"]
    evaluation_df = pd.DataFrame(bundle["evaluation_records"])
    save_r2_comparison(metrics)
    save_error_comparison(metrics)
    save_actual_vs_predicted(evaluation_df)
    save_feature_importance(bundle)
    save_symbolic_surfaces(bundle)
    save_complexity_loss(bundle)
    print(f"Updated result assets in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
