"""
08_hypothesis_testing.py

Día 3, Bloque 1: hipótesis del enunciado.

  Ejemplo 1: ¿el ticket promedio (monto_aplicado) es distinto entre
  APP y WEB? monto_aplicado no es normal (Bloque 2 del Día 2), así que
  el test principal es Mann-Whitney U (no paramétrico). Se reporta
  también t-test de Welch como comparación.

  Ejemplo 2: ¿el descuento (pct_descuento) afecta la cantidad comprada
  (unidades_producto_boleta)? Se usa correlación de Spearman con test
  de hipótesis formal (H0: rho=0) más una regresión simple. La
  correlación del Bloque 3 del Día 2 ya había dado r≈0.0002, p=0.74,
  así que se espera no encontrar efecto — se confirma formalmente acá.

Uso:
    python src/08_hypothesis_testing.py --lake data/lake_clean
"""
import argparse
import os

import duckdb
import numpy as np
from scipy import stats
import statsmodels.api as sm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lake", default="data/lake_clean", help="Data lake limpio (Parquet)")
    parser.add_argument(
        "--seed", type=int, default=int(os.environ.get("CPYD_SEED", 42)),
        help="Semilla fija (por defecto lee CPYD_SEED del entorno)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    print(f"[INFO] Semilla fija (CPYD_SEED): {args.seed}")

    glob_path = f"{args.lake}/**/*.parquet"
    con = duckdb.connect()
    df = con.sql(
        f"SELECT canal, monto_aplicado, pct_descuento, unidades_producto_boleta "
        f"FROM '{glob_path}'"
    ).df()
    print(f"[INFO] Filas cargadas: {len(df):,}")

    # -------------------------------------------------------------
    # Ejemplo 1: ticket promedio APP vs WEB
    # -------------------------------------------------------------
    print("\n=== Ejemplo 1: MONTO_APLICADO — APP vs WEB ===")
    print("H0: no hay diferencia entre APP y WEB | H1: sí hay diferencia")

    grupo_app = df.loc[df["canal"] == "APP", "monto_aplicado"]
    grupo_web = df.loc[df["canal"] == "WEB", "monto_aplicado"]
    print(
        f"APP: n={len(grupo_app):,}, media={grupo_app.mean():,.2f}, "
        f"mediana={grupo_app.median():,.2f}"
    )
    print(
        f"WEB: n={len(grupo_web):,}, media={grupo_web.mean():,.2f}, "
        f"mediana={grupo_web.median():,.2f}"
    )

    u_stat, p_mw = stats.mannwhitneyu(grupo_app, grupo_web, alternative="two-sided")
    print(f"\nMann-Whitney U (test principal): U={u_stat:.2f}, p={p_mw:.4g}")

    t_stat, p_t = stats.ttest_ind(grupo_app, grupo_web, equal_var=False)  # Welch
    print(f"t-test de Welch (comparación): t={t_stat:.4f}, p={p_t:.4g}")

    n1, n2 = len(grupo_app), len(grupo_web)
    r_efecto = 1 - (2 * u_stat) / (n1 * n2)
    print(f"Tamaño de efecto (r biserial de rango, aprox.): {r_efecto:.4f}")

    # -------------------------------------------------------------
    # Ejemplo 2: descuento vs cantidad
    # -------------------------------------------------------------
    print("\n=== Ejemplo 2: PCT_DESCUENTO vs UNIDADES_PRODUCTO_BOLETA ===")
    print("H0: rho=0 (sin relación) | H1: rho≠0")

    rho, p_rho = stats.spearmanr(df["pct_descuento"], df["unidades_producto_boleta"])
    print(f"Spearman: rho={rho:.4f}, p={p_rho:.4g}")

    X = sm.add_constant(df["pct_descuento"])
    modelo = sm.OLS(df["unidades_producto_boleta"], X).fit()
    print("\nRegresión simple (unidades_producto_boleta ~ pct_descuento):")
    print(modelo.summary().tables[1])
    print(f"R² = {modelo.rsquared:.5f}")


if __name__ == "__main__":
    main()