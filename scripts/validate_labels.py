"""
Validación de la correspondencia ASD/TC entre datos en disco y fenotipos.

Mapeo canónico del proyecto (fuente de verdad documentada):

    DX_GROUP (ABIDE)  →  carpeta   →  label numérico
    ----------------     --------     -------------
        1 (Autism)         ASD/            1
        2 (Control)        TC/             0

Es decir: **ASD = label 1, TC = label 0**.

El script cruza, para cada sujeto, tres fuentes que DEBEN coincidir:

  1. La carpeta donde quedó el PNG (`dataset/ASD/` o `dataset/TC/`), decidida por
     `downloader.py` al descargar según `CLASS_MAP`.
  2. El `label` que `dataset.discover_subjects()` le asigna al sujeto (lo que ve el
     modelo durante el entrenamiento).
  3. El `DX_GROUP` del CSV fenotípico de ABIDE (diagnóstico original, fuente de
     verdad independiente del pipeline de descarga).

Si alguna no coincide, los resultados del modelo (accuracy, ROC, etc.) podrían
estar invertidos. El script falla (exit 1) ante cualquier inconsistencia dura
para poder usarse como guard en la descarga o en CI.

Uso:
    uv run python scripts/validate_labels.py
    uv run python scripts/validate_labels.py --dataset-dir ./dataset \
        --meta-csv abide_data/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv --strict
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from scripts.dataset import discover_subjects

# ── Mapeo canónico (debe coincidir con downloader.CLASS_MAP y dataset.class_map) ──
DX_GROUP_TO_CLASS = {1: "ASD", 2: "TC"}      # idéntico a downloader.CLASS_MAP
CLASS_TO_LABEL = {"TC": 0, "ASD": 1}         # idéntico al class_map de discover_subjects
LABEL_TO_CLASS = {v: k for k, v in CLASS_TO_LABEL.items()}

DEFAULT_DATASET_DIR = "./dataset"
DEFAULT_META_CSV = "abide_data/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv"


def _load_dx_map(meta_csv: str) -> dict[str, int]:
    """Mapea SUB_ID (str) → DX_GROUP (int) desde el CSV fenotípico."""
    df = pd.read_csv(meta_csv, low_memory=False)
    if "SUB_ID" not in df.columns or "DX_GROUP" not in df.columns:
        raise ValueError(
            f"El CSV {meta_csv} no tiene columnas 'SUB_ID' y 'DX_GROUP'."
        )
    df = df.dropna(subset=["SUB_ID", "DX_GROUP"])
    return {str(int(row["SUB_ID"])): int(row["DX_GROUP"]) for _, row in df.iterrows()}


def validate_labels(
    dataset_dir: str = DEFAULT_DATASET_DIR,
    meta_csv: str = DEFAULT_META_CSV,
) -> dict:
    """
    Valida la correspondencia carpeta ↔ label ↔ DX_GROUP.

    Returns:
        dict con: 'n_subjects', 'n_ok', 'mismatches', 'duplicates',
        'missing_in_csv', 'invalid_dx', 'class_counts'.
    """
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"No existe el directorio de dataset: {dataset_dir}")
    if not os.path.exists(meta_csv):
        raise FileNotFoundError(f"No existe el CSV fenotípico: {meta_csv}")

    subjects = discover_subjects(dataset_dir, meta_csv)
    dx_map = _load_dx_map(meta_csv)

    mismatches: list[dict] = []      # label de carpeta ≠ label esperado por DX_GROUP
    duplicates: list[dict] = []      # mismo sujeto en ASD/ y TC/
    missing_in_csv: list[str] = []   # sujeto en disco sin fila en el CSV
    invalid_dx: list[dict] = []      # DX_GROUP fuera de {1, 2}
    n_ok = 0
    class_counts = {"ASD": 0, "TC": 0}
    seen_label: dict[str, int] = {}

    for s in subjects:
        sid = s["subject_id"]
        label = s["label"]
        folder = LABEL_TO_CLASS[label]
        class_counts[folder] += 1

        # Sujeto presente en ambas carpetas (labels contradictorios).
        if sid in seen_label and seen_label[sid] != label:
            duplicates.append({"subject_id": sid, "labels": sorted({seen_label[sid], label})})
        seen_label[sid] = label

        # ¿Está en el CSV? Sin DX_GROUP no podemos validar el diagnóstico.
        if sid not in dx_map:
            missing_in_csv.append(sid)
            continue

        dx = dx_map[sid]
        expected_class = DX_GROUP_TO_CLASS.get(dx)
        if expected_class is None:
            invalid_dx.append({"subject_id": sid, "dx_group": dx})
            continue

        expected_label = CLASS_TO_LABEL[expected_class]
        if expected_label != label:
            mismatches.append(
                {
                    "subject_id": sid,
                    "folder": folder,
                    "label": label,
                    "dx_group": dx,
                    "expected_class": expected_class,
                    "expected_label": expected_label,
                }
            )
        else:
            n_ok += 1

    return {
        "n_subjects": len(subjects),
        "n_ok": n_ok,
        "mismatches": mismatches,
        "duplicates": duplicates,
        "missing_in_csv": missing_in_csv,
        "invalid_dx": invalid_dx,
        "class_counts": class_counts,
    }


def _print_report(result: dict) -> None:
    print("=" * 60)
    print("Validación de labels ASD/TC")
    print("=" * 60)
    print("Mapeo canonico: DX_GROUP 1 -> ASD -> label 1 | DX_GROUP 2 -> TC -> label 0")
    print("-" * 60)
    cc = result["class_counts"]
    print(f"Sujetos descubiertos: {result['n_subjects']} (ASD={cc['ASD']}, TC={cc['TC']})")
    print(f"Sujetos validados OK: {result['n_ok']}")

    mismatches = result["mismatches"]
    duplicates = result["duplicates"]
    missing = result["missing_in_csv"]
    invalid = result["invalid_dx"]

    if mismatches:
        print(f"\n[ERROR] MISMATCHES ({len(mismatches)}): carpeta/label NO coincide con DX_GROUP")
        for m in mismatches:
            print(
                f"   sub {m['subject_id']}: esta en {m['folder']}/ (label={m['label']}) "
                f"pero DX_GROUP={m['dx_group']} => esperado {m['expected_class']} "
                f"(label={m['expected_label']})"
            )

    if duplicates:
        print(f"\n[ERROR] DUPLICADOS ({len(duplicates)}): mismo sujeto en ambas carpetas")
        for d in duplicates:
            print(f"   sub {d['subject_id']}: labels {d['labels']}")

    if invalid:
        print(f"\n[ERROR] DX_GROUP invalido ({len(invalid)}): fuera de {{1, 2}}")
        for i in invalid:
            print(f"   sub {i['subject_id']}: DX_GROUP={i['dx_group']}")

    if missing:
        print(f"\n[WARN] Sin fila en el CSV ({len(missing)}): no se pudo validar el diagnostico")
        preview = ", ".join(missing[:10])
        suffix = " …" if len(missing) > 10 else ""
        print(f"   {preview}{suffix}")

    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Valida la correspondencia ASD/TC entre carpetas, labels y DX_GROUP."
    )
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR, help="Directorio con ASD/ y TC/.")
    parser.add_argument("--meta-csv", default=DEFAULT_META_CSV, help="CSV fenotípico de ABIDE.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Falla (exit 1) también si hay sujetos sin fila en el CSV.",
    )
    args = parser.parse_args()

    result = validate_labels(args.dataset_dir, args.meta_csv)
    _print_report(result)

    hard_errors = (
        len(result["mismatches"])
        + len(result["duplicates"])
        + len(result["invalid_dx"])
    )
    if args.strict:
        hard_errors += len(result["missing_in_csv"])

    if hard_errors > 0:
        print(f"\nFALLÓ: {hard_errors} inconsistencia(s).")
        sys.exit(1)

    print("\nOK: todos los labels son consistentes con DX_GROUP.")


if __name__ == "__main__":
    main()
