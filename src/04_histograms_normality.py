"""
04_histograms_normality.py

Bloque 2 del Día 2: histogramas + curva de densidad + tests de
normalidad (Shapiro-Wilk y Lilliefors) para las variables numéricas
clave, y boxplots por categoría (CANAL y top 10 LOCAL por volumen).

Se usa Lilliefors en vez del KS estándar: como la media y desviación
se estiman de la misma muestra que se testea, el KS clásico da
p-valores inválidos (demasiado conservadores). Lilliefors corrige esto
(corrección de auditoría).

Los tests de normalidad se corren sobre una muestra aleatoria fija
(semilla CPYD_SEED) en vez de las 3.24M filas completas: con un N tan
grande, cualquier desviación mínima de la normalidad da p-valores
prácticamente cero (el test se vuelve hipersensible y deja de ser
informativo). Se documenta este criterio en el informe.

Uso:
    python src/04_histograms_normality.py --lake data/lake_clean --sample-size 5000
"""
import argparse
import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.diagnostic import lilliefors

VARIABLES_NUMERICAS = [
    "monto_aplicado",
    "edad",
    "pct_descuento",
    "frecuencia_compra_cliente",
    "unidades_producto_boleta",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lake", default="data/lake_clean", help="Data lake limpio (Parquet)")
    parser.add_argument("--output", default="reports/figures", help="Carpeta de salida de figuras")
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument(
        "--seed", type=int, default=int(os.environ.get("CPYD_SEED", 42)),
        help="Semilla fija (por defecto lee CPYD_SEED del entorno)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    print(f"[INFO] Semilla fija (CPYD_SEED): {args.seed}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    glob_path = f"{args.lake}/**/*.parquet"
    con = duckdb.connect()

    cols = ", ".join(VARIABLES_NUMERICAS + ["canal", "local"])
    print(f"[INFO] Cargando columnas necesarias desde '{args.lake}'...")
    df = con.sql(f"SELECT {cols} FROM '{glob_path}'").df()
    print(f"[INFO] Filas cargadas: {len(df):,}")

    # -------------------------------------------------------------
    # Tests de normalidad sobre muestra fija
    # -------------------------------------------------------------
    df_sample = df.sample(n=args.sample_size, random_state=args.seed)

    print(f"\n=== Tests de normalidad (muestra n={args.sample_size}, seed={args.seed}) ===")
    resultados = []
    for var in VARIABLES_NUMERICAS:
        datos = df_sample[var].dropna()
        shapiro_stat, shapiro_p = stats.shapiro(datos)
        # Lilliefors, no KS estándar: los parámetros (media, std) se
        # estiman de la MISMA muestra que se testea, lo cual invalida
        # las tablas críticas del KS clásico y da p-valores demasiado
        # conservadores. Lilliefors corrige esto (corrección de auditoría).
        lilliefors_stat, lilliefors_p = lilliefors(datos, dist="norm")
        resultados.append(
            {
                "variable": var,
                "shapiro_stat": round(shapiro_stat, 4),
                "shapiro_p": shapiro_p,
                "lilliefors_stat": round(lilliefors_stat, 4),
                "lilliefors_p": lilliefors_p,
                "es_normal_alpha_0.05": bool(shapiro_p > 0.05 and lilliefors_p > 0.05),
            }
        )
    resultados_df = pd.DataFrame(resultados)
    print(resultados_df.to_string(index=False))
    resultados_df.to_csv(output_dir / "tests_normalidad.csv", index=False)

    # -------------------------------------------------------------
    # Histogramas + densidad (sobre TODOS los datos, no la muestra)
    # -------------------------------------------------------------
    print("\n[INFO] Generando histogramas + densidad...")
    for var in VARIABLES_NUMERICAS:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df[var].dropna(), kde=True, stat="density", ax=ax, bins=50)
        ax.set_title(f"Distribución de {var}")
        fig.tight_layout()
        fig.savefig(output_dir / f"hist_{var}.png", dpi=150)
        plt.close(fig)

    # Versión log para monto_aplicado, dada su altísima asimetría (9.06)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(np.log1p(df["monto_aplicado"]), kde=True, stat="density", ax=ax, bins=50)
    ax.set_title("Distribución de log(monto_aplicado + 1)")
    fig.tight_layout()
    fig.savefig(output_dir / "hist_monto_aplicado_log.png", dpi=150)
    plt.close(fig)

    # -------------------------------------------------------------
    # Boxplots por categoría
    # -------------------------------------------------------------
    print("[INFO] Generando boxplots por categoría...")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="canal", y="monto_aplicado", ax=ax, showfliers=False)
    ax.set_title("MONTO_APLICADO por CANAL (outliers extremos ocultos para legibilidad)")
    fig.tight_layout()
    fig.savefig(output_dir / "boxplot_monto_por_canal.png", dpi=150)
    plt.close(fig)

    top_locales = df["local"].value_counts().head(10).index
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(
        data=df[df["local"].isin(top_locales)],
        x="local", y="monto_aplicado", ax=ax, showfliers=False,
    )
    ax.set_title("MONTO_APLICADO por LOCAL (top 10 por volumen de transacciones)")
    fig.tight_layout()
    fig.savefig(output_dir / "boxplot_monto_por_local.png", dpi=150)
    plt.close(fig)

    print(f"\n[OK] Figuras y resultados de tests guardados en '{output_dir}'")


if __name__ == "__main__":
    main()