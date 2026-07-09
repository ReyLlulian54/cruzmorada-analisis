"""
09_own_hypotheses.py

Día 3, Bloque 2: 3 hipótesis propias.

  H1: el monto promedio de compra (monto_aplicado) difiere según el
      género del cliente (GENERO 1 vs 2). Test: Mann-Whitney U
      (monto_aplicado no es normal, Bloque 2 del Día 2).

  H2: existe asociación entre la edad del cliente y su frecuencia de
      compra (visitas). Test: Spearman, calculado a NIVEL DE CLIENTE
      (una fila por codigo_cliente), no a nivel de transacción — si se
      calculara sobre todas las filas, un cliente con 97 compras
      pesaría 97 veces más que uno con 1 sola compra, sesgando la
      correlación hacia los clientes más frecuentes (pseudo-replicación).

  H3: el descuento promedio aplicado (PCT_DESCUENTO) varía según el
      canal de venta (CANAL). Test: Kruskal-Wallis (no paramétrico,
      pct_descuento no es normal).

Uso:
    python src/09_own_hypotheses.py --lake data/lake_clean
"""
import argparse
import os

import duckdb
import numpy as np
from scipy import stats


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
        f"SELECT codigo_cliente, canal, genero, edad, monto_aplicado, "
        f"pct_descuento, frecuencia_compra_cliente FROM '{glob_path}'"
    ).df()
    print(f"[INFO] Filas cargadas: {len(df):,}")

    # -------------------------------------------------------------
    # H1: monto_aplicado por GENERO
    # -------------------------------------------------------------
    print("\n=== H1: MONTO_APLICADO — genero 1 vs genero 2 ===")
    print("H0: no hay diferencia | H1: sí hay diferencia")
    g1 = df.loc[df["genero"] == 1, "monto_aplicado"]
    g2 = df.loc[df["genero"] == 2, "monto_aplicado"]
    print(f"Genero 1: n={len(g1):,}, media={g1.mean():,.2f}, mediana={g1.median():,.2f}")
    print(f"Genero 2: n={len(g2):,}, media={g2.mean():,.2f}, mediana={g2.median():,.2f}")

    u_stat, p_mw = stats.mannwhitneyu(g1, g2, alternative="two-sided")
    r_efecto = 1 - (2 * u_stat) / (len(g1) * len(g2))
    print(f"Mann-Whitney U: U={u_stat:.2f}, p={p_mw:.4g}, r efecto (aprox.)={r_efecto:.4f}")

    # -------------------------------------------------------------
    # H2: edad vs frecuencia_compra_cliente (a nivel de CLIENTE)
    # -------------------------------------------------------------
    print("\n=== H2: EDAD vs FRECUENCIA_COMPRA_CLIENTE (a nivel de cliente) ===")
    print("H0: rho=0 | H1: rho≠0")
    clientes = df.drop_duplicates(subset=["codigo_cliente"])
    print(f"Clientes únicos: {len(clientes):,}")

    rho, p_rho = stats.spearmanr(clientes["edad"], clientes["frecuencia_compra_cliente"])
    print(f"Spearman: rho={rho:.4f}, p={p_rho:.4g}")

    # -------------------------------------------------------------
    # H3: pct_descuento por CANAL
    # -------------------------------------------------------------
    print("\n=== H3: PCT_DESCUENTO por CANAL ===")
    print("H0: no hay diferencia entre canales | H1: sí hay diferencia")
    print(df.groupby("canal")["pct_descuento"].agg(["mean", "median", "std"]))

    grupos = [g["pct_descuento"].values for _, g in df.groupby("canal")]
    h_stat, p_val = stats.kruskal(*grupos)
    print(f"\nKruskal-Wallis: H={h_stat:.2f}, p={p_val:.4g}")


if __name__ == "__main__":
    main()