import argparse
import os
import urllib.request

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import datasets
from PIL import Image
from tqdm import tqdm

ABIDE_S3 = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs"
PIPELINE = "cpac"
STRATEGY = "filt_noglobal"
DEFAULT_DERIVATIVES = ("falff", "reho")
DEFAULT_OUTPUT_DIR = "./dataset"
DEFAULT_DATA_DIR = "./abide_data"
DEFAULT_MANIFEST_NAME = "manifest.csv"
CLASS_MAP = {1: "ASD", 2: "TC"}

# Índices de corte óptimos por derivado en espacio MNI 3mm (volumen 61×73×61).
# Provienen de statistical_analysis/vbm_analysis.py (Cohen's d ASD vs TC por plano,
# excluyendo TEST_SITES = [PITT, OLIN] para evitar data leakage en el slice selection).
SLICES = {
    "falff": {
        "sagital": lambda d: d[10, :, :],
        "coronal": lambda d: d[:, 8, :],
        "axial":   lambda d: d[:, :, 17],
    },
    "reho": {
        "sagital": lambda d: d[30, :, :],
        "coronal": lambda d: d[:, 10, :],
        "axial":   lambda d: d[:, :, 48],
    },
}


def to_png(slice_2d: np.ndarray, path: str) -> None:
    lo, hi = np.percentile(slice_2d, [1, 99])
    s = np.clip(slice_2d, lo, hi)
    s = (s - lo) / (hi - lo + 1e-8)
    Image.fromarray((s * 255).astype(np.uint8)).save(path)


def _download_nii(url: str, dest_path: str) -> None:
    """Descarga un .nii.gz mostrando progreso en MB. Saltea si ya existe."""
    if os.path.exists(dest_path):
        return
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    fname = os.path.basename(dest_path)
    with tqdm(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=f"    {fname}",
        leave=False,
        miniters=1,
    ) as pbar:

        def hook(blocks, block_size, total_size):
            if total_size > 0:
                pbar.total = total_size
            pbar.update(blocks * block_size - pbar.n)

        urllib.request.urlretrieve(url, dest_path, reporthook=hook)


def _process_subject(
    sub_id: str,
    file_id: str,
    class_name: str,
    nii_dir: str,
    output_dir: str,
    derivatives: tuple[str, ...],
) -> list[str]:
    """Procesa sujeto, devuelve lista de paths PNG generados (relativos a output_dir)."""
    out_dir = os.path.join(output_dir, class_name)
    os.makedirs(out_dir, exist_ok=True)
    pngs: list[str] = []

    for deriv in derivatives:
        nii_path = os.path.join(nii_dir, f"{file_id}_{deriv}.nii.gz")
        url = f"{ABIDE_S3}/{PIPELINE}/{STRATEGY}/{deriv}/{file_id}_{deriv}.nii.gz"

        _download_nii(url, nii_path)

        data = nib.load(nii_path).get_fdata()
        for view, extractor in SLICES[deriv].items():
            png_name = f"{sub_id}_{deriv}_{view}.png"
            to_png(extractor(data), os.path.join(out_dir, png_name))
            pngs.append(os.path.join(class_name, png_name))
    return pngs


def _get_subjects(data_dir: str) -> pd.DataFrame:
    """Lee CSV fenotípico completo de ABIDE PCP. Mantiene TODAS las columnas.

    Filtros aplicados:
      - FILE_ID != "no_filename" (descarta sujetos sin volumen disponible).
      - SUB_ID, FILE_ID, DX_GROUP no nulos (claves para procesamiento).

    Filtros QC (movimiento, raters): NO se aplican acá. Política del proyecto es
    descargar todo y decidir el threshold post-hoc vía scripts/qc_eda.py, que
    marca keep_for_train en el manifest.

    El resto de columnas (SITE_ID, AGE_AT_SCAN, SEX, FIQ/VIQ/PIQ, ADOS_*, ADI_*,
    func_mean_fd, qc_*_rater_*, etc.) se conservan para persistir en el manifest.
    NO se inyectan al modelo.
    """
    pheno_path = os.path.join(data_dir, "ABIDE_pcp", "Phenotypic_V1_0b_preprocessed1.csv")

    if not os.path.exists(pheno_path):
        # Usa nilearn solo para descargar el CSV de fenotipos (descarga 1 sujeto como bootstrap)
        print("Descargando metadata de ABIDE...")
        datasets.fetch_abide_pcp(
            data_dir=data_dir,
            pipeline=PIPELINE,
            band_pass_filtering=True,
            global_signal_regression=False,
            derivatives=["falff"],
            n_subjects=1,
            quality_checked=True,
            verbose=0,
        )

    df = pd.read_csv(pheno_path)
    df = df[df["FILE_ID"] != "no_filename"].copy()
    df = df.dropna(subset=["SUB_ID", "FILE_ID", "DX_GROUP"]).reset_index(drop=True)
    return df


def _apply_max_subjects(
    subjects: pd.DataFrame, max_subjects: int | None, subset_seed: int
) -> pd.DataFrame:
    """Submuestra reproducible (sin estratificar por clase; para smoke tests)."""
    if max_subjects is None or len(subjects) <= max_subjects:
        return subjects
    return subjects.sample(n=max_subjects, random_state=subset_seed).reset_index(drop=True)


def _write_manifest(
    rows: list[dict], pheno_df: pd.DataFrame, output_dir: str, manifest_name: str
) -> str:
    """Persiste manifest cruzando filas procesadas con CSV fenotípico completo.

    Salida: dataset/manifest.csv con TODAS las columnas fenotípicas + columnas extra
    de tracking del procesamiento. Source of truth para splits por sitio, balance
    demográfico y auditoría. NO debe ser leído como features por el modelo.
    """
    if not rows:
        return ""
    proc_df = pd.DataFrame(rows)
    # Cruce por SUB_ID (clave canónica). FILE_ID también único pero SUB_ID es el
    # prefijo del PNG → cruce simétrico al que necesita train.
    pheno_indexed = pheno_df.set_index("SUB_ID")
    proc_df["SUB_ID"] = proc_df["SUB_ID"].astype(int)
    merged = proc_df.merge(
        pheno_indexed,
        left_on="SUB_ID",
        right_index=True,
        how="left",
        suffixes=("", "_pheno"),
    )
    manifest_path = os.path.join(output_dir, manifest_name)
    os.makedirs(output_dir, exist_ok=True)
    merged.to_csv(manifest_path, index=False)
    return manifest_path


def run(
    data_dir: str = DEFAULT_DATA_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    derivatives: tuple[str, ...] = DEFAULT_DERIVATIVES,
    max_subjects: int | None = None,
    subset_seed: int = 42,
    manifest_name: str = DEFAULT_MANIFEST_NAME,
) -> None:
    subjects = _get_subjects(data_dir)
    subjects = _apply_max_subjects(subjects, max_subjects, subset_seed)
    nii_dir = os.path.join(data_dir, "ABIDE_pcp", PIPELINE, STRATEGY)

    if max_subjects is not None:
        print(f"Subconjunto: {len(subjects)} sujetos (max_subjects={max_subjects}, seed={subset_seed})")

    errors = 0
    processed_rows: list[dict] = []
    for _, row in tqdm(subjects.iterrows(), total=len(subjects), desc="Sujetos"):
        sub_id = str(int(row["SUB_ID"]))
        file_id = str(row["FILE_ID"])
        class_name = CLASS_MAP.get(int(row["DX_GROUP"]), "unknown")

        try:
            pngs = _process_subject(sub_id, file_id, class_name, nii_dir, output_dir, derivatives)
            processed_rows.append(
                {
                    "SUB_ID": sub_id,
                    "FILE_ID": file_id,
                    "class_name": class_name,
                    "derivatives": ";".join(derivatives),
                    "png_paths": ";".join(pngs),
                    "n_pngs": len(pngs),
                    "status": "ok",
                }
            )
        except Exception as e:
            errors += 1
            tqdm.write(f"  SKIP {file_id}: {e}")
            processed_rows.append(
                {
                    "SUB_ID": sub_id,
                    "FILE_ID": file_id,
                    "class_name": class_name,
                    "derivatives": ";".join(derivatives),
                    "png_paths": "",
                    "n_pngs": 0,
                    "status": f"error: {type(e).__name__}",
                }
            )

    ok = len(subjects) - errors
    print(f"\n{ok}/{len(subjects)} sujetos procesados. Dataset en: {os.path.abspath(output_dir)}")

    manifest_path = _write_manifest(processed_rows, subjects, output_dir, manifest_name)
    if manifest_path:
        print(f"Manifest fenotípico (todas columnas) escrito en: {manifest_path}")
        print("  Uso recomendado: splits por SITE_ID, balance demográfico, auditoría QC.")
        print("  NO inyectar columnas fenotípicas al modelo (el CNN recibe solo PNG).")


def _parse_download_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Descarga derivados ABIDE CPAC (fALFF/ReHo) y genera PNGs por corte."
    )
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Caché ABIDE / CSV fenotípico")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Salida ASD/ y TC/")
    p.add_argument(
        "--deriv",
        action="append",
        dest="derivs",
        choices=tuple(SLICES.keys()),
        help="Derivado a procesar (repetible; por defecto falff y reho).",
    )
    p.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        metavar="N",
        help="Procesar solo N sujetos (muestra aleatoria reproducible con --subset-seed).",
    )
    p.add_argument(
        "--subset-seed",
        type=int,
        default=42,
        help="Semilla para la submuestra cuando --max-subjects está definido.",
    )
    p.add_argument(
        "--manifest-name",
        default=DEFAULT_MANIFEST_NAME,
        help="Nombre del CSV manifest dentro de --output-dir (default: manifest.csv).",
    )
    return p.parse_args()


def main_cli() -> None:
    args = _parse_download_args()
    derivs = tuple(args.derivs) if args.derivs else DEFAULT_DERIVATIVES
    run(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        derivatives=derivs,
        max_subjects=args.max_subjects,
        subset_seed=args.subset_seed,
        manifest_name=args.manifest_name,
    )


if __name__ == "__main__":
    main_cli()
