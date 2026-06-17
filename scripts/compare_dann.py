import argparse

from scripts.train import run_from_config


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Entrena y compara dos modelos con DANN y sin DANN. "
            "Ambos tienen la misma arquitectura (feature extractor + class classifier + "
            "domain classifier). En el modelo sin DANN, alpha=0.0 anula los gradientes "
            "del GRL hacia el feature extractor, desactivando la adaptacion de dominio "
            "sin alterar la cantidad de parametros. La comparacion es controlada."
        )
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Ruta a config TOML compartido por ambos modelos (ej. config.toml).",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Comparación: DANN vs SIN DANN")
    print("=" * 60)

    print("\n[1/2] Entrenando modelo SIN DANN (alpha=0.0)")
    run_from_config(config_path=args.config, mode="no-dann", output_dir="./results_no_dann")

    print("\n[2/2] Entrenando modelo CON DANN")
    run_from_config(config_path=args.config, mode="dann", output_dir="./results_dann")

    print("\n" + "=" * 60)
    print("Comparación completada.")
    print("  SIN DANN: ./results_no_dann")
    print("  CON DANN: ./results_dann")
    print("  Revisá classification_report.txt, confusion_matrix.png y roc_curve.png en cada carpeta.")
    print("=" * 60)


if __name__ == "__main__":
    main()
