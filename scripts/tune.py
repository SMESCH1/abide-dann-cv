"""
Búsqueda de hiperparámetros DANN con Optuna + MLflow.

Ejecución estándar (overnight ~6-8h):
    uv run python scripts/tune.py

Variables de entorno opcionales:
    OPTUNA_N_TRIALS         Cantidad de trials (default 500).
    OPTUNA_EPOCHS_PER_TRIAL Epochs máximos por trial (default 60).
    OPTUNA_STUDY_NAME       Nombre del study (default 'asd_dann_v1'). Permite
                            resumir un study previo (SQLite storage).
    OPTUNA_TIMEOUT_HOURS    Corta el optimize si se cumple (default sin tope).
    OPTUNA_PROFILE          'standard' o 'aggressive' para explorar más rápido.

Persiste en SQLite (`results/optuna_study.db`) → reanudable si crashea.
Trackea cada trial en MLflow (`./mlruns`, experimento ASD_DANN_HyperTuning).
Artefactos por trial en `results/trials/trial_XXXX/` (sin pisado entre corridas).

Cada vez que un trial supera al mejor previo, vuelca los best_params a
`results/best_params.toml` (no espera al final). Ese archivo es independiente
de `config.toml`: copialo/mergealo a mano cuando quieras entrenar el final.

Pruner MedianPruner: corta trials cuyo AUC reportado en epoch k es peor que
la mediana de trials previos. Ahorra cómputo en zonas pobres del espacio.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from scripts.train import run_training

DEFAULT_N_TRIALS = int(os.environ.get("OPTUNA_N_TRIALS", "500"))
DEFAULT_EPOCHS = int(os.environ.get("OPTUNA_EPOCHS_PER_TRIAL", "50"))
STUDY_NAME = os.environ.get("OPTUNA_STUDY_NAME", "asd_dann_v1")
TIMEOUT_HOURS = float(os.environ.get("OPTUNA_TIMEOUT_HOURS", "0"))
PROFILE = os.environ.get("OPTUNA_PROFILE", "standard").strip().lower()
STORAGE_DIR = "./results_tune"
STORAGE_URL = f"sqlite:///{STORAGE_DIR}/optuna_study.db"
BEST_PARAMS_PATH = os.path.join(STORAGE_DIR, "best_params.toml")

# Claves que se vuelcan al TOML, en el orden en que aparecen en config.toml.
_TUNED_KEYS = ("lr", "batch_size", "dann_alpha", "dropout", "label_smoothing", "patience", "scheduler_patience")


def _toml_value(value) -> str:
    """Formatea un valor Python como literal TOML (float con precisión completa)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    return f'"{value}"'


def _write_best_params_toml(study: optuna.Study) -> None:
    """Vuelca los best_params actuales a BEST_PARAMS_PATH (escritura atómica)."""
    params = study.best_params
    lines = [
        "# best_params.toml — generado automáticamente por tune.py.",
        f"# Trial {study.best_trial.number}  AUC={study.best_value:.4f}  "
        f"profile={PROFILE}  {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "# Copialo/mergealo a config.toml para entrenar el modelo final.",
        "# Faltan epochs/seed (no se tunean): tomalos de config.toml.",
        "",
    ]
    lines += [f"{k} = {_toml_value(params[k])}" for k in _TUNED_KEYS if k in params]

    tmp_path = BEST_PARAMS_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp_path, BEST_PARAMS_PATH)


def _save_best_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
    """Reescribe el TOML solo cuando el trial recién terminado es el nuevo mejor."""
    if trial.state != optuna.trial.TrialState.COMPLETE:
        return
    if study.best_trial.number != trial.number:
        return
    _write_best_params_toml(study)
    print(f"  ↳ best_params.toml actualizado (AUC={study.best_value:.4f})")


def objective(trial: optuna.Trial) -> float:
    # Espacio de búsqueda ampliado y perfil agresivo opcional.
    if PROFILE == "aggressive":
        lr = trial.suggest_float("lr", 1e-6, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        dann_alpha = trial.suggest_float("dann_alpha", 0.0, 1.5)
        dropout = trial.suggest_float("dropout", 0.0, 0.75)
        label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.35)
        patience = trial.suggest_int("patience", 6, 16)
        scheduler_patience = trial.suggest_int("scheduler_patience", 2, 8)
    else:
        lr = trial.suggest_float("lr", 5e-6, 5e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        dann_alpha = trial.suggest_float("dann_alpha", 0.0, 1.0)
        dropout = trial.suggest_float("dropout", 0, 0.1)
        label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.25)
        
        patience = 25 #trial.suggest_int("patience", 15, 25)
        scheduler_patience = 10 #trial.suggest_int("scheduler_patience", 10, 12)

    trial_output_dir = os.path.join(STORAGE_DIR, "trials", f"trial_{trial.number:04d}")

    with mlflow.start_run(nested=True):
        mlflow.log_params(
            {
                "lr": lr,
                "batch_size": batch_size,
                "dann_alpha": dann_alpha,
                "dropout": dropout,
                "label_smoothing": label_smoothing,
                "patience": patience,
                "scheduler_patience": scheduler_patience,
                "epochs_cap": DEFAULT_EPOCHS,
                "optuna_profile": PROFILE,
                "trial_number": trial.number,
                "evaluate_test": False,
                "trial_output_dir": trial_output_dir,
            }
        )

        print(
            f"\n--- Trial {trial.number} ---\n"
            f"lr={lr:.1e}, bs={batch_size}, alpha={dann_alpha:.2f}, "
            f"dropout={dropout:.2f}, smoothing={label_smoothing:.2f}, "
            f"patience={patience}, scheduler_patience={scheduler_patience}"
        )

        best_auc = run_training(
            batch_size=batch_size,
            lr=lr,
            dann_alpha=dann_alpha,
            dropout=dropout,
            label_smoothing=label_smoothing,
            epochs=DEFAULT_EPOCHS,
            patience=patience,
            scheduler_patience=scheduler_patience,
            trial=trial,
            output_dir=trial_output_dir,
            evaluate_test=False,
        )

        mlflow.log_metric("best_auc", best_auc)

    return best_auc


def main() -> None:
    os.makedirs(STORAGE_DIR, exist_ok=True)
    mlflow.set_tracking_uri("./mlruns")
    mlflow.set_experiment("ASD_DANN_HyperTuning")

    if PROFILE not in {"standard", "aggressive"}:
        raise ValueError("OPTUNA_PROFILE debe ser 'standard' o 'aggressive'")

    sampler = TPESampler(
        seed=42,
        n_startup_trials=8 if PROFILE == "aggressive" else 15,
        multivariate=True,
    )
    pruner = MedianPruner(
        n_startup_trials=6 if PROFILE == "aggressive" else 10,
        n_warmup_steps=3 if PROFILE == "aggressive" else 5,
        interval_steps=1,
    )

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_URL,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    timeout_seconds = TIMEOUT_HOURS * 3600 if TIMEOUT_HOURS > 0 else None

    print(
        f"Optuna study='{STUDY_NAME}' | storage={STORAGE_URL}\n"
        f"  profile: {PROFILE}\n"
        f"  trials objetivo: {DEFAULT_N_TRIALS}  epochs cap: {DEFAULT_EPOCHS}  "
        f"timeout: {TIMEOUT_HOURS}h ({timeout_seconds}s)\n"
        f"  trials previos: {len(study.trials)}"
    )

    started = time.time()
    with mlflow.start_run(run_name=f"Optuna_{STUDY_NAME}"):
        study.optimize(
            objective,
            n_trials=DEFAULT_N_TRIALS,
            timeout=timeout_seconds,
            gc_after_trial=True,
            callbacks=[_save_best_callback],
        )

        elapsed = time.time() - started
        completed = sum(
            1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        )
        pruned = sum(
            1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
        )
        print("\n" + "=" * 60)
        print("Búsqueda completada.")
        print(
            f"  Trials totales: {len(study.trials)}  completados: {completed}  "
            f"pruned: {pruned}"
        )
        print(f"  Tiempo total: {elapsed/3600:.2f}h")
        if completed == 0:
            print("  No hubo trials completados (todos podados/fallidos).")
            mlflow.log_metric("n_completed_trials", completed)
            mlflow.log_metric("n_pruned_trials", pruned)
            mlflow.log_metric("elapsed_hours", elapsed / 3600)
            return
        print(f"  Mejor AUC: {study.best_value:.4f}")
        print("  Parámetros:")
        for key, value in study.best_params.items():
            print(f"    {key}: {value}")

        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("overall_best_auc", study.best_value)
        mlflow.log_metric("n_completed_trials", completed)
        mlflow.log_metric("n_pruned_trials", pruned)
        mlflow.log_metric("elapsed_hours", elapsed / 3600)

    print(f"\nPara ver los resultados en MLflow, ejecuta: uv run mlflow ui")
    print(f"Storage Optuna: {STORAGE_URL}")


if __name__ == "__main__":
    main()
