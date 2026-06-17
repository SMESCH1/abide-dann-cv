"""
Genera la tabla comparativa de resultados (DANN vs sin DANN) en Markdown,
leyéndola directamente desde los `metrics_summary.json` de cada modelo.

Así la tabla del README no tiene números hardcodeados: se regenera desde los
artefactos reales de cada corrida.

Uso:
    uv run python scripts/report.py                  # imprime la tabla por stdout
    uv run python scripts/report.py -o RESULTS.md    # además la escribe a un archivo
"""

import argparse
import json
import sys
from pathlib import Path

# Métricas a mostrar: (clave en test_metrics, etiqueta legible, decimales)
TEST_METRICS = [
    ("auc", "ROC AUC", 3),
    ("balanced_accuracy", "Balanced accuracy", 3),
    ("accuracy", "Accuracy", 3),
    ("f1", "F1 (ASD)", 3),
]


def load_metrics(results_dir: Path) -> dict:
    path = results_dir / "metrics_summary.json"
    if not path.exists():
        sys.exit(
            f"No se encontró {path}. Corré primero "
            f"`uv run python scripts/compare_dann.py --config config.toml`."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fmt(value: float, decimals: int) -> str:
    return f"{value:.{decimals}f}"


def build_table(no_dann: dict, dann: dict) -> str:
    rows = ["| Metric (held-out sites) | Without DANN | With DANN |",
            "|---|:---:|:---:|"]

    for key, label, dec in TEST_METRICS:
        a = fmt(no_dann["test_metrics"][key], dec)
        b = fmt(dann["test_metrics"][key], dec)
        rows.append(f"| {label} | {a} | **{b}** |")

    # AUC por sitio (los dos sitios reservados como test ciego)
    sites = sorted(dann.get("test_auc_by_site", {}))
    for site in sites:
        a = fmt(no_dann["test_auc_by_site"][site], 3)
        b = fmt(dann["test_auc_by_site"][site], 3)
        rows.append(f"| AUC @ {site} | {a} | **{b}** |")

    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dann-dir", default="results_dann", type=Path)
    parser.add_argument("--no-dann-dir", default="results_no_dann", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Archivo Markdown opcional donde escribir la tabla.")
    args = parser.parse_args()

    dann = load_metrics(args.dann_dir)
    no_dann = load_metrics(args.no_dann_dir)

    table = build_table(no_dann, dann)
    print(table)

    if args.output is not None:
        args.output.write_text(table + "\n", encoding="utf-8")
        print(f"\nTabla escrita en {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
