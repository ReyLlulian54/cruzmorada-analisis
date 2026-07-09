"""
05_association_analysis.py

Bloque 3 del Día 2: análisis de asociación entre variables.

  - Matriz de correlación de Spearman con p-values (no Pearson, ya
    que ninguna variable pasó los tests de normalidad del Bloque 2).
  - Chi-cuadrado CANAL vs LOCAL. Se corre la versión completa (792
    locales) y se reporta el % de celdas con frecuencia esperada < 5
    (diagnóstico de violación de supuestos por la alta cardinalidad
    de LOCAL), y una versión agregada (top 10 locales + "OTROS") más
    robusta, con Cramér's V como medida de tamaño de efecto.
  - Kruskal-Wallis (no paramétrico, equivalente a ANOVA) para
    MONTO_APLICADO por CANAL y por LOCAL (top 10).

Uso:
    python src/05_association_analysis.py --lake data/lake_clean
"""
import argparse
import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg

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
    parser.add_argument("--output", default="reports/figures", help="Carpeta de salida")
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
    # Matriz de correlación (Spearman) con p-values
    # -------------------------------------------------------------
    print("\n=== Correlaciones de Spearman (pares, con p-values) ===")
    corr_result = pg.pairwise_corr(df[VARIABLES_NUMERICAS], method="spearman")
    print(corr_result[["X", "Y", "r", "p_unc"]].to_string(index=False))
    corr_result.to_csv(output_dir / "correlaciones_spearman.csv", index=False)

    corr_matrix = df[VARIABLES_NUMERICAS].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, ax=ax, fmt=".2f")
    ax.set_title("Matriz de correlación (Spearman)")
    fig.tight_layout()
    fig.savefig(output_dir / "matriz_correlacion_spearman.png", dpi=150)
    plt.close(fig)

    # -------------------------------------------------------------
    # Chi-cuadrado: CANAL vs LOCAL
    # -------------------------------------------------------------
    print("\n=== Chi-cuadrado: CANAL vs LOCAL (versión completa, 792 locales) ===")
    tabla_completa = pd.crosstab(df["canal"], df["local"])
    chi2, p, dof, expected = stats.chi2_contingency(tabla_completa)
    pct_celdas_bajas = (expected < 5).mean() * 100
    print(f"chi2={chi2:.2f}, p={p:.4g}, gl={dof}")
    print(
        f"[AVISO] {pct_celdas_bajas:.1f}% de las celdas tienen frecuencia esperada < 5 "
        f"(supuesto del chi-cuadrado violado por la alta cardinalidad de LOCAL)"
    )

    print("\n=== Chi-cuadrado: CANAL vs LOCAL (versión agregada: top 10 + OTROS) ===")
    top_locales = df["local"].value_counts().head(10).index
    df["local_agrupado"] = np.where(
        df["local"].isin(top_locales), df["local"].astype(str), "OTROS"
    )
    tabla_agrupada = pd.crosstab(df["canal"], df["local_agrupado"])
    chi2_a, p_a, dof_a, expected_a = stats.chi2_contingency(tabla_agrupada)
    pct_celdas_bajas_a = (expected_a < 5).mean() * 100
    n_total = tabla_agrupada.values.sum()
    cramers_v = np.sqrt(chi2_a / (n_total * (min(tabla_agrupada.shape) - 1)))
    print(f"chi2={chi2_a:.2f}, p={p_a:.4g}, gl={dof_a}, Cramér's V={cramers_v:.4f}")
    print(f"[INFO] {pct_celdas_bajas_a:.1f}% de las celdas con frecuencia esperada < 5")

    # -------------------------------------------------------------
    # Kruskal-Wallis (no paramétrico): MONTO_APLICADO por CANAL / LOCAL
    # -------------------------------------------------------------
    print("\n=== Kruskal-Wallis: MONTO_APLICADO por CANAL ===")
    grupos_canal = [g["monto_aplicado"].values for _, g in df.groupby("canal")]
    h_stat, p_val = stats.kruskal(*grupos_canal)
    print(f"H={h_stat:.2f}, p={p_val:.4g}")

    print("\n=== Kruskal-Wallis: MONTO_APLICADO por LOCAL (top 10) ===")
    df_top = df[df["local"].isin(top_locales)]
    grupos_local = [g["monto_aplicado"].values for _, g in df_top.groupby("local")]
    h_stat_l, p_val_l = stats.kruskal(*grupos_local)
    print(f"H={h_stat_l:.2f}, p={p_val_l:.4g}")

    print(f"\n[OK] Resultados y figuras guardados en '{output_dir}'")


if __name__ == "__main__":
    main()