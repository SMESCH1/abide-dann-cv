"""
Evaluación y visualización de resultados del clasificador ASD vs TC.

Genera:
- Confusion matrix
- Curva ROC
- Curvas de entrenamiento (loss y accuracy)
- Reporte de clasificación
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

RESULTS_DIR = "./results"
CLASS_NAMES = ["TC", "ASD"]


def plot_training_curves(history: dict, save_dir: str) -> None:
    """Genera gráficos de loss y accuracy durante el entrenamiento."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], "r-", label="Val", linewidth=2)
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss durante entrenamiento")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], "b-", label="Train", linewidth=2)
    axes[1].plot(epochs, history["val_acc"], "r-", label="Val", linewidth=2)
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy durante entrenamiento")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # AUC
    axes[2].plot(epochs, history["val_auc"], "g-", label="Val AUC", linewidth=2)
    axes[2].set_xlabel("Época")
    axes[2].set_ylabel("AUC-ROC")
    axes[2].set_title("AUC-ROC en validación")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Curvas de entrenamiento: {path}")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_dir: str) -> None:
    """Genera y guarda la confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix — Test Set", fontsize=14)
    path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix: {path}")


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, save_dir: str) -> None:
    """Genera y guarda la curva ROC."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, "b-", linewidth=2, label=f"ROC (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Curva ROC — Test Set", fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    path = os.path.join(save_dir, "roc_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Curva ROC: {path}")


def run_evaluation(results_dir: str = RESULTS_DIR) -> None:
    """Genera todos los gráficos y reportes."""
    print("\n" + "=" * 60)
    print(f"Generando visualizaciones para: {results_dir}")
    print("=" * 60)

    # Cargar historial y summary para obtener el threshold optimizado
    history_path = os.path.join(results_dir, "history.json")
    with open(history_path) as f:
        history = json.load(f)

    summary_path = os.path.join(results_dir, "metrics_summary.json")
    selected_threshold = 0.5
    try:
        with open(summary_path) as f:
            summary = json.load(f)
        selected_threshold = summary.get("threshold_selection", {}).get("selected_threshold", 0.5)
    except FileNotFoundError:
        print(f"  Advertencia: No se encontró {summary_path}. Usando threshold=0.5 por defecto.")

    # Cargar predicciones del test
    preds_path = os.path.join(results_dir, "test_predictions.npz")
    data = np.load(preds_path)
    y_true = data["y_true"]
    y_proba = data["y_proba"]
    y_pred = (y_proba > selected_threshold).astype(int)
    sites = data.get("sites", np.array([]))

    # Gráficos
    plot_training_curves(history, results_dir)
    plot_confusion_matrix(y_true, y_pred, results_dir)  # Ahora usa el threshold correcto
    plot_roc_curve(y_true, y_proba, results_dir)

    # Reporte de clasificación
    print("\n" + "=" * 60)
    print("Reporte de Clasificación — Test Set")
    print(f"(usando threshold optimizado = {selected_threshold:.3f})")
    print("-" * 60)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0)
    print(report)

    # Guardar reporte
    report_path = os.path.join(results_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("Reporte de Clasificación — Test Set\n")
        f.write(f"(usando threshold optimizado = {selected_threshold:.3f})\n")
        f.write("=" * 50 + "\n\n")
        f.write(report)
    print(f"  Reporte guardado: {report_path}")


def main():
    import sys
    res_dir = sys.argv[1] if len(sys.argv) > 1 else RESULTS_DIR
    run_evaluation(res_dir)

if __name__ == "__main__":
    main()
