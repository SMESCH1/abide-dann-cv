"""
Pipeline de entrenamiento para el clasificador ASD vs TC.

- Split estratificado por sujeto (70/15/15)
- Entrenamiento con early stopping
- Guarda el mejor modelo según val AUC
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import os
import time
import tomllib
from typing import Any

import mlflow
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler

from scripts.dataset import SubjectDataset, discover_subjects
from scripts.model import ASD_DANN, count_parameters

# ── Hiperparámetros ──────────────────────────────────────────────
DATASET_DIR = "./dataset"
OUTPUT_DIR = "./results"
BATCH_SIZE = 64
EPOCHS = 300
LR = 0.0002073644811030639
PATIENCE = 30
SEED = 42
TEST_SITES = ["PITT", "OLIN"]  # Sitios excluidos para evaluar generalización
DANN_ALPHA = 0.1
DROPOUT = 0.3
LABEL_SMOOTHING = 0.1
SCHEDULER_PATIENCE = 20
# ─────────────────────────────────────────────────────────────────

class _BCEWithLabelSmoothing(nn.Module):
    """BCE binaria con label smoothing simétrico.

    Suaviza los targets: 1 → (1 - smoothing/2), 0 → smoothing/2.
    Regulariza la confianza del modelo sin alterar el balance de clases.
    """
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets_smooth = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(inputs, targets_smooth)


def set_seed(seed: int) -> None:
    """Fijar semillas para reproducibilidad."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """AUC robusta cuando hay una sola clase presente."""
    try:
        return float(roc_auc_score(y_true, y_proba))
    except ValueError:
        return 0.0


def _safe_pr_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """PR-AUC robusta cuando hay una sola clase presente."""
    try:
        return float(average_precision_score(y_true, y_proba))
    except ValueError:
        return 0.0


def compute_binary_metrics(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Métricas estándar para clasificación binaria."""
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "auc": _safe_auc(y_true, y_proba),
        "pr_auc": _safe_pr_auc(y_true, y_proba),
        "accuracy": float((y_pred == y_true).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }


def select_best_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> tuple[float, dict[str, float], dict[str, float]]:
    """
    Selecciona threshold en validación maximizando balanced accuracy.
    Desempata por F1.
    Retorna (best_threshold, val_metrics_best, val_metrics_at_0_5).
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 91)

    metrics_at_05 = compute_binary_metrics(y_true, y_proba, threshold=0.5)
    best_threshold = 0.5
    best_metrics = metrics_at_05
    best_score = metrics_at_05["balanced_accuracy"]
    best_f1 = metrics_at_05["f1"]

    for th in thresholds:
        m = compute_binary_metrics(y_true, y_proba, threshold=float(th))
        score = m["balanced_accuracy"]
        f1 = m["f1"]
        if (score > best_score) or (score == best_score and f1 > best_f1):
            best_threshold = float(th)
            best_metrics = m
            best_score = score
            best_f1 = f1

    return best_threshold, best_metrics, metrics_at_05


def sanitize_metric_name(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name).strip("_").lower()


def compute_site_auc(y_true: np.ndarray, y_proba: np.ndarray, sites: np.ndarray) -> dict[str, float]:
    """AUC por sitio cuando hay ambas clases presentes."""
    site_metrics: dict[str, float] = {}
    for site in sorted(set(sites.tolist())):
        mask = sites == site
        y_site = y_true[mask]
        if len(np.unique(y_site)) < 2:
            continue
        site_metrics[str(site)] = _safe_auc(y_site, y_proba[mask])
    return site_metrics


def split_subjects(
    subjects: list[dict],
    test_sites: list[str] = TEST_SITES,
    val_size: float = 0.15,
    seed: int = SEED,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split estratificado dejando test_sites exclusivamente en el test set."""
    test = [s for s in subjects if s["site"] in test_sites]
    train_val = [s for s in subjects if s["site"] not in test_sites]

    # Verificar que no hay overlap entre test y train/val
    train_val_sites = set(s["site"] for s in train_val)
    overlap = set(test_sites) & train_val_sites
    if overlap:
        raise ValueError(f"Data leakage: sitios {overlap} aparecen en test y train/val")

    # Estratificar por clase Y sitio para evitar que sitios chicos queden
    # 100% en train o 100% en val, lo que sesgaría el early stopping.
    strata = [f"{s['label']}_{s['site']}" for s in train_val]
    from collections import Counter
    counts = Counter(strata)
    if any(count < 2 for count in counts.values()):
        print("  [WARN] Algunas combinaciones de clase y sitio tienen solo 1 sujeto. Estratificando solo por clase (label) para evitar error.")
        strata = [s["label"] for s in train_val]
    train, val = train_test_split(
        train_val, test_size=val_size, stratify=strata, random_state=seed
    )

    return train, val, test


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    domain_criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    site_to_idx: dict,
    alpha: float = DANN_ALPHA,
) -> tuple[float, float]:
    """Entrena una época con DANN. Retorna (class_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y, sites in loader:
        x, y = x.to(device), y.to(device)
        
        # Mapear strings de sitios a índices numéricos
        unknown = [s for s in sites if s not in site_to_idx]
        if unknown:
            raise ValueError(f"Sitios desconocidos en batch: {set(unknown)}. Conocidos: {list(site_to_idx)}")
        site_indices = [site_to_idx[s] for s in sites]
        site_labels = torch.tensor(site_indices, dtype=torch.long, device=device)

        optimizer.zero_grad()
        
        # Forward pass (devuelve logits de clase y logits de dominio)
        class_logits, domain_logits = model(x, alpha=alpha)
        class_logits = class_logits.squeeze(1)
        
        # Calcular pérdidas
        class_loss = criterion(class_logits, y)
        domain_loss = domain_criterion(domain_logits, site_labels)
        
        # Implementación canónica DANN (Ganin et al. 2016): loss = L_task + L_domain,
        # donde la ponderación -alpha se aplica en el GRL durante backprop.
        # No ponderar domain_loss acá; hacerlo introduciría alpha dos veces (alpha²).
        loss = class_loss + domain_loss
        loss.backward()
        optimizer.step()

        total_loss += class_loss.item() * x.size(0)
        preds = (torch.sigmoid(class_logits) > 0.5).float()
        correct += (preds == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Evalúa el modelo. Retorna (loss, accuracy, auc, y_true, y_proba, sites)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_probs = []
    all_sites = []

    for x, y, site in loader:
        x, y = x.to(device), y.to(device)
        class_logits, _ = model(x)
        logits = class_logits.squeeze(1)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        correct += (preds == y).sum().item()
        total += x.size(0)

        all_labels.extend(y.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_sites.extend(site)

    y_true = np.array(all_labels)
    y_proba = np.array(all_probs)
    sites = np.array(all_sites)

    try:
        auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        auc = 0.0

    return total_loss / total, correct / total, auc, y_true, y_proba, sites


def run_training(
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS,
    lr: float = LR,
    patience: int = PATIENCE,
    dann_alpha: float = DANN_ALPHA,
    dropout: float = DROPOUT,
    label_smoothing: float = LABEL_SMOOTHING,
    seed: int = SEED,
    scheduler_patience: int = SCHEDULER_PATIENCE,
    output_dir: str = OUTPUT_DIR,
    trial: optuna.Trial | None = None,
    evaluate_test: bool = True,
    calibrate_threshold: bool = True,
) -> float:
    """Pipeline completo de entrenamiento."""
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Descubrir sujetos ──
    print("\nCargando dataset...")
    subjects = discover_subjects(DATASET_DIR)
    labels = [s["label"] for s in subjects]
    n_tc = labels.count(0)
    n_asd = labels.count(1)
    print(f"  Sujetos totales: {len(subjects)} (ASD={n_asd}, TC={n_tc})")

    # ── Split ──
    train_subj, val_subj, test_subj = split_subjects(subjects)
    print(f"  Train: {len(train_subj)}, Val: {len(val_subj)}, Test: {len(test_subj)}")

    # Guardar IDs del test set para reproducibilidad
    test_ids = [s["subject_id"] for s in test_subj]
    with open(os.path.join(output_dir, "test_subjects.json"), "w") as f:
        json.dump(test_ids, f)

    # ── Datasets & Loaders ──
    train_ds = SubjectDataset(train_subj, augment=True)
    val_ds = SubjectDataset(val_subj, augment=False)
    test_ds = SubjectDataset(test_subj, augment=False)

    train_labels = [s["label"] for s in train_subj]
    class_counts = [train_labels.count(0), train_labels.count(1)]
    sample_weights = [1.0 / class_counts[int(l)] for l in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_labels), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # ── Mapa de Sitios de Entrenamiento ──
    train_sites = sorted(list(set(s["site"] for s in train_subj + val_subj)))
    site_to_idx = {site: idx for idx, site in enumerate(train_sites)}
    print(f"\nSitios en entrenamiento: {train_sites}")
    
    # ── Modelo ──
    model = ASD_DANN(num_sites=len(train_sites), dropout=dropout).to(device)
    print(f"Parámetros entrenables: {count_parameters(model):,}")

    # WeightedRandomSampler ya balancea los batches (~50/50 ASD/TC),
    # por lo que BCE con label smoothing es suficiente. Focal Loss con
    # alpha != 0.5 conflictuaría con el sampler al re-ponderar clases ya balanceadas.
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=None,
        reduction="mean",
    ) if label_smoothing == 0.0 else _BCEWithLabelSmoothing(smoothing=label_smoothing)
    domain_criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=scheduler_patience
    )

    # ── Entrenamiento ──
    print("\n" + "=" * 60)
    print("Iniciando entrenamiento")
    print("=" * 60)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_auc": [], "val_pr_auc": [],
    }

    best_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(output_dir, "best_model.pth")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, domain_criterion, optimizer, device, site_to_idx, alpha=dann_alpha
        )

        # Validate
        val_loss, val_acc, val_auc, y_true_val, y_proba_val, _ = evaluate(
            model, val_loader, criterion, device
        )
        val_pr_auc = _safe_pr_auc(y_true_val, y_proba_val)

        if trial is not None:
            trial.report(val_auc, step=epoch)
            if trial.should_prune():
                print(f"\n! Trial podado por Optuna en epoch {epoch} (val_auc={val_auc:.4f})")
                raise optuna.TrialPruned()

        # Scheduler
        scheduler.step(val_auc)

        # Guardar métricas
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)
        history["val_pr_auc"].append(val_pr_auc)

        # Log
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  AUC: {val_auc:.4f}  PR-AUC: {val_pr_auc:.4f} | "
            f"LR: {current_lr:.1e}"
        )

        if mlflow.active_run() is not None:
            mlflow.log_metric("epoch_train_loss", train_loss, step=epoch)
            mlflow.log_metric("epoch_train_acc", train_acc, step=epoch)
            mlflow.log_metric("epoch_val_loss", val_loss, step=epoch)
            mlflow.log_metric("epoch_val_acc", val_acc, step=epoch)
            mlflow.log_metric("epoch_val_auc", val_auc, step=epoch)
            mlflow.log_metric("epoch_val_pr_auc", val_pr_auc, step=epoch)
            mlflow.log_metric("epoch_lr", current_lr, step=epoch)

        # Early stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  > Mejor modelo guardado (AUC={best_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n! Early stopping en epoch {epoch} (patience={patience})")
                break

    elapsed = time.time() - start_time
    print(f"\nEntrenamiento completado en {elapsed:.1f}s")
    print(f"Mejor Val AUC: {best_auc:.4f}")

    # Guardar historial
    history_serializable = {k: [float(v) for v in vals] for k, vals in history.items()}
    with open(os.path.join(output_dir, "history.json"), "w") as f:
        json.dump(history_serializable, f, indent=2)

    if evaluate_test:
        # ── Evaluación final en test ──
        print("\n" + "=" * 60)
        print("Evaluación en Test Set")
        print("=" * 60)

        model.load_state_dict(torch.load(best_model_path, weights_only=True))
        _, _, _, y_true_val, y_proba_val, _ = evaluate(
            model, val_loader, criterion, device
        )
        test_loss, test_acc, test_auc, y_true, y_proba, test_sites = evaluate(
            model, test_loader, criterion, device
        )

        selected_threshold = 0.5
        val_metrics_at_05 = compute_binary_metrics(y_true_val, y_proba_val, threshold=0.5)
        val_metrics_selected = val_metrics_at_05
        if calibrate_threshold:
            selected_threshold, val_metrics_selected, val_metrics_at_05 = select_best_threshold(
                y_true_val, y_proba_val
            )

        test_metrics_at_05 = compute_binary_metrics(y_true, y_proba, threshold=0.5)
        test_metrics = compute_binary_metrics(y_true, y_proba, threshold=selected_threshold)
        site_auc = compute_site_auc(y_true, y_proba, test_sites)

        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Accuracy: {test_acc:.4f}")
        print(f"  Test AUC: {test_auc:.4f}")
        print(f"  Threshold seleccionado: {selected_threshold:.2f}")
        print(f"  Test PR-AUC: {test_metrics['pr_auc']:.4f}")
        print(f"  Test F1: {test_metrics['f1']:.4f}")
        print(f"  Test Recall: {test_metrics['recall']:.4f}")
        print(f"  Test Specificity: {test_metrics['specificity']:.4f}")

        # Guardar predicciones del test
        np.savez(
            os.path.join(output_dir, "test_predictions.npz"),
            y_true=y_true,
            y_proba=y_proba,
            sites=test_sites,
        )

        metrics_summary: dict[str, Any] = {
            "best_val_auc": float(best_auc),
            "test_loss": float(test_loss),
            "threshold_selection": {
                "selected_threshold": float(selected_threshold),
                "criterion": "max_balanced_accuracy_then_f1_on_validation",
                "calibrate_threshold": bool(calibrate_threshold),
            },
            "val_metrics_threshold_0_5": val_metrics_at_05,
            "val_metrics_selected_threshold": val_metrics_selected,
            "test_metrics_threshold_0_5": test_metrics_at_05,
            "test_metrics": test_metrics,
            "test_auc_by_site": site_auc,
        }
        with open(os.path.join(output_dir, "metrics_summary.json"), "w") as f:
            json.dump(metrics_summary, f, indent=2)

        if mlflow.active_run() is not None:
            mlflow.log_metric("best_val_auc", best_auc)
            mlflow.log_metric("test_loss", test_loss)
            mlflow.log_metric("selected_threshold", float(selected_threshold))
            mlflow.log_metric("val_balanced_accuracy_selected_threshold", val_metrics_selected["balanced_accuracy"])
            mlflow.log_metric("val_f1_selected_threshold", val_metrics_selected["f1"])
            for metric_name, metric_value in test_metrics.items():
                mlflow.log_metric(f"test_{metric_name}", metric_value)
            for metric_name, metric_value in test_metrics_at_05.items():
                mlflow.log_metric(f"test_t05_{metric_name}", metric_value)
            for site_name, site_value in site_auc.items():
                mlflow.log_metric(f"test_auc_site_{sanitize_metric_name(site_name)}", site_value)
    else:
        print("\nEvaluación en test omitida (modo tuning).")

    print(f"\nResultados guardados en: {os.path.abspath(output_dir)}")
    return best_auc


def load_config(config_path: str | None) -> dict[str, Any]:
    """Lee un config TOML externo. Devuelve {} si no se pasa ruta."""
    if not config_path:
        return {}
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def run_from_config(
    config_path: str | None = None,
    mode: str = "dann",
    output_dir: str | None = None,
    do_eval: bool = True,
) -> float:
    """
    Entrena un modelo resolviendo hiperparámetros desde un config externo.

    Los valores ausentes en el config caen a los defaults del módulo.
    mode="dann" usa `dann_alpha` del config; mode="no-dann" fuerza alpha=0.0.

    Nota sobre la comparación: ambos modos usan la misma arquitectura (feature
    extractor + class classifier + domain classifier). Con alpha=0.0 el GRL
    anula todos los gradientes de la rama de dominio hacia el feature extractor
    (grad * -0.0 = 0), por lo que funcionalmente es equivalente a no tener DANN.
    La arquitectura idéntica garantiza igual cantidad de parámetros y capacidad,
    haciendo la comparación controlada.
    """
    if mode not in ("dann", "no-dann"):
        raise ValueError("mode debe ser 'dann' o 'no-dann'")

    cfg = load_config(config_path)
    dann_alpha = float(cfg.get("dann_alpha", DANN_ALPHA)) if mode == "dann" else 0.0
    if output_dir is None:
        output_dir = "./results_dann" if mode == "dann" else "./results_no_dann"

    best_auc = run_training(
        batch_size=int(cfg.get("batch_size", BATCH_SIZE)),
        epochs=int(cfg.get("epochs", EPOCHS)),
        lr=float(cfg.get("lr", LR)),
        patience=int(cfg.get("patience", PATIENCE)),
        dann_alpha=dann_alpha,
        dropout=float(cfg.get("dropout", DROPOUT)),
        label_smoothing=float(cfg.get("label_smoothing", LABEL_SMOOTHING)),
        seed=int(cfg.get("seed", SEED)),
        scheduler_patience=int(cfg.get("scheduler_patience", SCHEDULER_PATIENCE)),
        output_dir=output_dir,
    )

    if do_eval:
        from scripts.evaluate import run_evaluation
        run_evaluation(output_dir)

    return best_auc


def main():
    parser = argparse.ArgumentParser(
        description="Entrena un modelo ASD vs TC, con o sin DANN."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Ruta a config TOML (ej. config.toml). Si se omite, usa los defaults del módulo.",
    )
    parser.add_argument(
        "--mode",
        choices=["dann", "no-dann"],
        default="dann",
        help="'dann' usa dann_alpha del config; 'no-dann' fuerza alpha=0.0.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directorio de salida. Default: ./results_dann o ./results_no_dann según --mode.",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="No generar gráficos (curvas, matriz de confusión, ROC) al finalizar.",
    )
    args = parser.parse_args()

    run_from_config(
        config_path=args.config,
        mode=args.mode,
        output_dir=args.output_dir,
        do_eval=not args.no_eval,
    )


if __name__ == "__main__":
    main()
