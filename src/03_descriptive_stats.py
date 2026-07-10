"""

Estadística descriptiva completa sobre el dataset limpio (data/lake_clean),
usando DuckDB (funciones de agregación estadística nativas: skewness,
kurtosis, stddev, quantiles) directamente sobre los archivos Parquet.

Uso:
    python src/03_descriptive_stats.py --lake data/lake_clean
"""
import argparse
import os
import duckdb

VARIABLES_NUMERICAS = [
    "monto_aplicado",
    "edad",
    "pct_descuento",
    "frecuencia_compra_cliente",
    "unidades_producto_boleta",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lake", default="data/lake_clean", help="Data lake limpio (Parquet)")
    args = parser.parse_args()

    if not os.path.exists(args.lake):
        raise FileNotFoundError(f"El data lake especificado no existe: {args.lake}")

    glob_path = f"{args.lake}/**/*.parquet"
    con = duckdb.connect()

    for var in VARIABLES_NUMERICAS:
        print(f"\n=== Estadística descriptiva: {var} ===")
        print(
            con.sql(
                f"""
                SELECT
                    COUNT({var})                    AS n,
                    ROUND(AVG({var}), 4)             AS media,
                    ROUND(MEDIAN({var}), 4)          AS mediana,
                    ROUND(STDDEV({var}), 4)          AS desv_std,
                    ROUND(MIN({var}), 4)             AS minimo,
                    ROUND(MAX({var}), 4)             AS maximo,
                    ROUND(QUANTILE_CONT({var}, 0.25), 4) AS q1,
                    ROUND(QUANTILE_CONT({var}, 0.75), 4) AS q3,
                    ROUND(SKEWNESS({var}), 4)        AS asimetria,
                    ROUND(KURTOSIS({var}), 4)        AS curtosis
                FROM '{glob_path}'
                """
            ).df()
        )

    print("\n=== Estadística descriptiva por CANAL (monto_aplicado) ===")
    print(
        con.sql(
            f"""
            SELECT canal,
                   COUNT(*) AS n,
                   ROUND(AVG(monto_aplicado), 2) AS media_monto,
                   ROUND(MEDIAN(monto_aplicado), 2) AS mediana_monto,
                   ROUND(STDDEV(monto_aplicado), 2) AS desv_std_monto
            FROM '{glob_path}'
            GROUP BY canal
            ORDER BY n DESC
            """
        ).df()
    )


if __name__ == "__main__":
    main()