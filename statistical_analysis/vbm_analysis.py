import os
import json
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import ttest_ind
from tqdm import tqdm
import matplotlib.pyplot as plt

DATA_DIR = "../abide_data"
PIPELINE = "cpac"
STRATEGY = "filt_noglobal"
DERIVATIVES = ["falff", "reho"]
OUTPUT_DIR = "./"

CLASS_MAP = {1: "ASD", 2: "TC"}
TEST_SITES = ["PITT", "OLIN"]

# Umbral de background: vóxeles con media < este percentil del volumen se excluyen de la máscara.
MASK_PERCENTILE = 10

# Selección robusta de cortes (evita planos de borde con estimaciones ruidosas):
#   (a) BORDER_FRACTION: excluye este % de planos en cada extremo de cada eje.
#   (b) MIN_MASK_FRACTION: un plano debe tener al menos esta fracción de los vóxeles
#       cerebrales del plano más lleno de su eje para ser elegible.
BORDER_FRACTION = 0.1
MIN_MASK_FRACTION = 0.2


def get_subjects(data_dir: str) -> pd.DataFrame:
    pheno_path = os.path.join(data_dir, "ABIDE_pcp", "Phenotypic_V1_0b_preprocessed1.csv")
    if not os.path.exists(pheno_path):
        raise FileNotFoundError(f"No se encontró {pheno_path}.")

    df = pd.read_csv(pheno_path, low_memory=False)
    df = df[df["FILE_ID"] != "no_filename"].dropna(subset=["SUB_ID", "FILE_ID", "DX_GROUP"])
    if "SITE_ID" in df.columns:
        df = df[~df["SITE_ID"].isin(TEST_SITES)]
    return df[["SUB_ID", "FILE_ID", "DX_GROUP"]].reset_index(drop=True)


def load_volumes(subjects_df: pd.DataFrame, nii_dir: str, derivative: str):
    asd_vols, tc_vols = [], []
    print(f"\nCargando volúmenes para: {derivative}...")
    for _, row in tqdm(subjects_df.iterrows(), total=len(subjects_df)):
        nii_path = os.path.join(nii_dir, f"{row['FILE_ID']}_{derivative}.nii.gz")
        if not os.path.exists(nii_path):
            continue
        try:
            data = nib.load(nii_path).get_fdata().astype(np.float32)
            if int(row["DX_GROUP"]) == 1:
                asd_vols.append(data)
            elif int(row["DX_GROUP"]) == 2:
                tc_vols.append(data)
        except Exception as e:
            print(f"Error cargando {nii_path}: {e}")
    return np.array(asd_vols), np.array(tc_vols)


def build_brain_mask(asd_vols: np.ndarray, tc_vols: np.ndarray) -> np.ndarray:
    """Máscara booleana: vóxeles con media global > percentil MASK_PERCENTILE."""
    all_vols = np.concatenate([asd_vols, tc_vols], axis=0)
    mean_vol = np.mean(all_vols, axis=0)
    threshold = np.percentile(mean_vol, MASK_PERCENTILE)
    return mean_vol > threshold


def cohens_d_volume(asd_vols: np.ndarray, tc_vols: np.ndarray) -> np.ndarray:
    """Cohen's d vóxel a vóxel: (mean_ASD - mean_TC) / pooled_std."""
    n1, n2 = asd_vols.shape[0], tc_vols.shape[0]
    mu1 = np.mean(asd_vols, axis=0)
    mu2 = np.mean(tc_vols, axis=0)
    var1 = np.var(asd_vols, axis=0, ddof=1)
    var2 = np.var(tc_vols, axis=0, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        d = (mu1 - mu2) / pooled_std
    return np.nan_to_num(d, nan=0.0)


def best_slice_per_axis(
    effect_volume: np.ndarray,
    mask: np.ndarray,
    border_fraction: float = BORDER_FRACTION,
    min_mask_fraction: float = MIN_MASK_FRACTION,
) -> dict:
    """
    Para cada eje (x=sagital, y=coronal, z=axial) puntúa cada plano candidato
    como la media de |Cohen's d| dentro de la máscara cerebral y devuelve el índice óptimo.

    Selección robusta para evitar artefactos de borde:
      (a) Excluye los `border_fraction` de planos en cada extremo del eje.
      (b) Descalifica planos cuya máscara cerebral tenga menos del `min_mask_fraction`
          de los vóxeles del plano más lleno del eje (estimaciones de pocos vóxeles
          son ruidosas e inflan la media de |d|).
    Los planos inelegibles reciben score -inf, por lo que `argmax` nunca los elige.
    """
    abs_d = np.abs(effect_volume)
    results = {}

    axes = {"sagital": 0, "coronal": 1, "axial": 2}
    for view, ax in axes.items():
        n_slices = effect_volume.shape[ax]
        margin = int(round(border_fraction * n_slices))
        mask_counts = np.array(
            [int(np.take(mask, i, axis=ax).sum()) for i in range(n_slices)]
        )
        min_voxels = max(1, int(min_mask_fraction * mask_counts.max()))

        scores = np.full(n_slices, -np.inf)
        for i in range(n_slices):
            if i < margin or i >= n_slices - margin:
                continue  # (a) plano de borde
            if mask_counts[i] < min_voxels:
                continue  # (b) máscara demasiado chica
            plane_d = np.take(abs_d, i, axis=ax)
            plane_m = np.take(mask, i, axis=ax)
            scores[i] = float(plane_d[plane_m].mean())

        if not np.isfinite(scores).any():
            raise ValueError(
                f"Ningún plano elegible en eje '{view}' "
                f"(margin={margin}, min_voxels={min_voxels}). Revisá los umbrales."
            )

        best_idx = int(np.argmax(scores))
        results[view] = {
            "index": best_idx,
            "mean_abs_d": float(scores[best_idx]),
            "mask_voxels": int(mask_counts[best_idx]),
        }

    return results


def save_tmap_slice(effect_volume: np.ndarray, coords: dict, derivative: str):
    x = coords["sagital"]["index"]
    y = coords["coronal"]["index"]
    z = coords["axial"]["index"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Cohen's d — {derivative.upper()}  (x={x}, y={y}, z={z})", fontsize=14)

    vmax = np.nanpercentile(np.abs(effect_volume), 99)
    vmin = -vmax

    for ax_obj, data_slice, title in [
        (axes[0], np.rot90(effect_volume[x, :, :]), f"Sagital (x={x})"),
        (axes[1], np.rot90(effect_volume[:, y, :]), f"Coronal (y={y})"),
        (axes[2], np.rot90(effect_volume[:, :, z]), f"Axial (z={z})"),
    ]:
        im = ax_obj.imshow(data_slice, cmap="coolwarm", vmin=vmin, vmax=vmax)
        ax_obj.set_title(title)

    fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.05, label="Cohen's d")
    plt.savefig(os.path.join(OUTPUT_DIR, f"optimal_tmap_{derivative}.png"), dpi=150)
    plt.close()


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    data_dir = os.path.abspath(DATA_DIR)
    nii_dir = os.path.join(data_dir, "ABIDE_pcp", PIPELINE, STRATEGY)

    subjects_df = get_subjects(data_dir)
    print(f"Sujetos encontrados (sin sitios test): {len(subjects_df)}")

    results = {}

    for derivative in DERIVATIVES:
        asd_vols, tc_vols = load_volumes(subjects_df, nii_dir, derivative)
        print(f"  ASD: {len(asd_vols)}  TC: {len(tc_vols)}")

        if len(asd_vols) == 0 or len(tc_vols) == 0:
            print(f"Sin datos suficientes para {derivative}. Saltando.")
            continue

        mask = build_brain_mask(asd_vols, tc_vols)
        print(f"  Vóxeles en máscara cerebral: {mask.sum()}")

        d_vol = cohens_d_volume(asd_vols, tc_vols)

        # t-test para guardar p-value del vóxel pico (informativo)
        with np.errstate(invalid="ignore", divide="ignore"):
            t_stat, p_val = ttest_ind(asd_vols, tc_vols, axis=0, equal_var=False)
        p_val = np.nan_to_num(p_val, nan=1.0)
        peak_idx = np.unravel_index(np.argmin(p_val), p_val.shape)

        coords = best_slice_per_axis(d_vol, mask)

        results[derivative] = {
            "optimal_slice_x_sagital": coords["sagital"]["index"],
            "optimal_slice_y_coronal": coords["coronal"]["index"],
            "optimal_slice_z_axial":   coords["axial"]["index"],
            "mean_abs_d_sagital": coords["sagital"]["mean_abs_d"],
            "mean_abs_d_coronal": coords["coronal"]["mean_abs_d"],
            "mean_abs_d_axial":   coords["axial"]["mean_abs_d"],
            "mask_voxels_sagital": coords["sagital"]["mask_voxels"],
            "mask_voxels_coronal": coords["coronal"]["mask_voxels"],
            "mask_voxels_axial":   coords["axial"]["mask_voxels"],
            "peak_voxel_p_value": float(p_val[peak_idx]),
            "peak_voxel_t_stat":  float(t_stat[peak_idx]),
            "selection_params": {
                "border_fraction": BORDER_FRACTION,
                "min_mask_fraction": MIN_MASK_FRACTION,
                "mask_percentile": MASK_PERCENTILE,
            },
        }

        print(f"  Cortes óptimos {derivative}:")
        for view in ("sagital", "coronal", "axial"):
            k = coords[view]["index"]
            score = coords[view]["mean_abs_d"]
            print(f"    {view}: idx={k}  mean|d|={score:.4f}")

        print("  Generando visualización...")
        save_tmap_slice(d_vol, coords, derivative)

        del asd_vols, tc_vols, t_stat, p_val, d_vol

    out_json = os.path.join(OUTPUT_DIR, "optimal_slices.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResultados guardados en {os.path.abspath(out_json)}")


if __name__ == "__main__":
    main()
