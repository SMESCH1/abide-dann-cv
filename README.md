# Autism (ASD) classification from rs-fMRI as a Computer Vision problem — CNN + adversarial domain generalization (DANN)

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![uv](https://img.shields.io/badge/built%20with-uv-de5fe9.svg)](https://github.com/astral-sh/uv)
[![CI](https://github.com/SMESCH1/TP_CV/actions/workflows/ci.yml/badge.svg)](https://github.com/SMESCH1/TP_CV/actions/workflows/ci.yml)

> 🇪🇸 **Versión en español:** [README.es.md](README.es.md)

Binary classification of **Autism Spectrum Disorder (ASD)** vs. **typical controls (TC)** from resting-state fMRI, reframed as a **computer-vision** task: instead of functional-connectivity matrices, we turn voxel-wise derived maps (**fALFF** and **ReHo**) from the [ABIDE](http://fcon_1000.projects.nitrc.org/indi/abide/) consortium into **2D grayscale slices** and train a small **CNN**. To fight the multi-site scanner bias (*site effects*), we add a **Domain-Adversarial Neural Network (DANN)** so the feature extractor learns site-invariant representations and generalizes to **unseen acquisition sites**.

> ⚕️ **Academic project — not a diagnostic tool.** Built for a Computer Vision graduate course (M.Sc. in Artificial Intelligence). The goal is a reproducible end-to-end pipeline and an honest evaluation of adversarial domain adaptation, *not* clinical use.

---

## Key results

Two identical models (same hyperparameters, same data) differing **only** in the domain-adversarial term. The test set is two **entire sites held out** (`PITT` + `OLIN`, 90 subjects) never seen in training — a true out-of-distribution generalization test.

| Metric (held-out sites) | Without DANN | **With DANN** |
|---|:---:|:---:|
| ROC AUC | 0.552 | **0.588** |
| Balanced accuracy | 0.504 *(≈ chance)* | **0.570** |
| Accuracy | 0.478 | **0.556** |
| F1 (ASD) | 0.175 | **0.459** |
| AUC @ OLIN | 0.568 | **0.618** |
| AUC @ PITT | 0.527 | **0.557** |

**Takeaway:** the no-DANN model collapses toward chance on unseen scanners (balanced accuracy ≈ 0.50), while DANN keeps a **small but consistent edge** across both held-out sites. The absolute numbers are modest *by design* — see [Honest results & limitations](#honest-results--limitations).

<p align="center">
  <img src="results_dann/roc_curve.png" alt="ROC curve (DANN, test set)" width="45%"/>
  <img src="results_dann/confusion_matrix.png" alt="Confusion matrix (DANN, test set)" width="45%"/>
</p>

---

## Approach

1. **CV reframing.** We consume 3D parametric maps (fALFF, ReHo) already registered to **MNI152** space, not raw 4D BOLD or connectomes.
2. **Statistically-selected slices.** For each derivative and axis we pick the orthogonal slice that maximizes the voxel-wise Cohen's *d* between groups — computed **only on training subjects** to avoid leakage. Each subject becomes a **6-channel** tensor (2 derivatives × 3 views).
3. **Small CNN** exploiting local spatial patterns.
4. **DANN** with a Gradient Reversal Layer: a domain classifier over the 18 training sites pushes the feature extractor toward site-invariant features (multi-source adversarial *domain generalization*).
5. **Leakage-safe splits:** by subject *and* by site (whole sites held out for test).

Full methodology: [docs/contexto_proyecto.md](docs/contexto_proyecto.md) · slice criteria: [docs/cortes_mni_y_mapas_derivados.md](docs/cortes_mni_y_mapas_derivados.md) · final report: [docs/INFORME_FINAL.md](docs/INFORME_FINAL.md) · references: [docs/referencias.md](docs/referencias.md).

---

## Quickstart

Built with [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

**1 — Generate the dataset (downloads ABIDE derivatives → 2D PNGs):**

```bash
uv run python main.py                              # full phenotypic table
uv run python main.py --max-subjects 20 --subset-seed 42   # quick smoke test
```

**2 — Train (CNN + DANN). Hyperparameters live in a single [config.toml](config.toml):**

```bash
uv run python scripts/train.py --config config.toml --mode dann      # with DANN
uv run python scripts/train.py --config config.toml --mode no-dann   # baseline (alpha=0)
```

**3 — Reproduce the head-to-head comparison:**

```bash
uv run python scripts/compare_dann.py --config config.toml
```

**Optional — hyperparameter search (Optuna + MLflow):**

```bash
uv run python scripts/tune.py    # study persists to SQLite, resumable; trials logged to ./mlruns
```

Each run writes metrics and figures (training curves, confusion matrix, ROC, classification report) to `results_dann/` or `results_no_dann/`.

---

## Repository layout

```
main.py                 # dataset generation (ABIDE NIfTI → 2D PNG slices)
config.toml             # shared hyperparameters for both models
scripts/
  downloader.py         # fetch ABIDE derivatives, extract slices
  dataset.py            # subject-grouped dataset & splits
  model.py              # CNN + DANN (Gradient Reversal Layer)
  train.py              # training + evaluation + figures
  compare_dann.py       # DANN vs no-DANN head-to-head
  tune.py               # Optuna hyperparameter search
  evaluate.py           # standalone evaluation
  validate_labels.py    # label-integrity check (folder vs CSV vs model)
statistical_analysis/   # VBM-like optimal slice selection (Cohen's d)
results_dann/ , results_no_dann/   # metrics + figures per model
docs/                   # technical context, references, final report
paper/                  # IEEEtran LaTeX write-up
```

---

## Reproducibility

- **Single source of hyperparameters:** [config.toml](config.toml) is shared by both models; the only difference is the DANN term (`dann_alpha`, forced to `0` for the baseline).
- **Fixed seed:** `seed = 42` controls splits and initialization, so runs are repeatable.
- **Regenerate the figures & metrics:** `uv run python scripts/compare_dann.py --config config.toml` re-trains both models and rewrites the artifacts in `results_dann/` and `results_no_dann/`.
- **Regenerate the results table above:** `uv run python scripts/report.py` reads the `metrics_summary.json` files and prints the Markdown table — the numbers in this README are not hand-typed.

## Honest results & limitations

This problem is **genuinely hard** and the README reflects that on purpose:

- The winning slices have **|Cohen's d| ≈ 0.07–0.09** — conventionally "negligible". The selection method is sound; the underlying signal is weak, which is consistent with the known difficulty of subject-level ASD vs TC discrimination in rs-fMRI.
- Compressing 3D float volumes to 8-bit PNG slices **loses dynamic range** (a deliberate trade-off for a 2D-CNN prototype; the cached NIfTI remains the source of truth).
- Reporting a model that lands near chance without DANN, and only modestly above it with DANN, is the point: the contribution is **measuring whether adversarial domain generalization helps across unseen sites**, not chasing a headline accuracy.

---

## Team & attribution

Group project for **Visión y Percepción Computarizada**, M.Sc. in Artificial Intelligence.

- Sebastián Mesch Henriques · [@SMESCH1](https://github.com/SMESCH1)
- Leandro Carcagno · [@lcgno](https://github.com/lcgno)

This repository is a portfolio fork of the original team repository. Code is MIT-licensed; the **ABIDE dataset and its CPAC derivatives are governed by their own terms of use** — see the [ABIDE](http://fcon_1000.projects.nitrc.org/indi/abide/) and Preprocessed Connectomes Project documentation.
