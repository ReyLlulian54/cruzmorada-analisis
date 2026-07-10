"""

Detección explícita de outliers con métodos robustos (IQR y Z-score)
sobre las variables numéricas continuas, tal como pide el enunciado.

Esto es un diagnóstico ADICIONAL a las reglas de limpieza basadas en
dominio ya aplicadas en 02_cleaning_and_features.py (PCT_DESCUENTO
fuera de [0,1], EDAD fuera de [0,100] -> tratados ahí como errores de
registro). Acá no se elimina ni modifica nada: se reporta cuántos
puntos caen fuera de los límites de IQR y cuántos tienen |Z|>3, para
documentar en el informe si son errores de registro o casos de
negocio reales.

Los outliers de
MONTO_APLICADO detectados acá muy probablemente NO son errores —
según el modelo de regresión (script 10), el monto está fuertemente
asociado al nivel de descuento (que a su vez actúa como proxy del
precio/categoría del producto), así que montos muy altos son
consistentes con productos caros reales, no con datos corruptos.

Uso:
    python src/11_outlier_detection.py --lake data/lake_clean
"""
import argparse

import duckdb
import pandas as pd

VARIABLES = ["monto_aplicado", "edad", "pct_descuento", "frecuencia_compra_cliente"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lake", default="data/lake_clean", help="Data lake limpio (Parquet)")
    return parser.parse_args()


def main():
    args = parse_args()
    glob_path = f"{args.lake}/**/*.parquet"
    con = duckdb.connect()
    df = con.sql(f"SELECT {', '.join(VARIABLES)} FROM '{glob_path}'").df()
    n_total = len(df)
    print(f"[INFO] Filas cargadas: {n_total:,}")

    resultados = []
    for var in VARIABLES:
        datos = df[var].dropna()

        q1, q3 = datos.quantile(0.25), datos.quantile(0.75)
        iqr = q3 - q1
        lim_inf, lim_sup = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers_iqr = int(((datos < lim_inf) | (datos > lim_sup)).sum())

        z = (datos - datos.mean()) / datos.std()
        outliers_z = int((z.abs() > 3).sum())

        resultados.append(
            {
                "variable": var,
                "q1": round(q1, 4),
                "q3": round(q3, 4),
                "limite_inf_iqr": round(lim_inf, 4),
                "limite_sup_iqr": round(lim_sup, 4),
                "outliers_iqr": outliers_iqr,
                "pct_iqr": round(100 * outliers_iqr / n_total, 3),
                "outliers_zscore": outliers_z,
                "pct_zscore": round(100 * outliers_z / n_total, 3),
            }
        )

    tabla = pd.DataFrame(resultados)
    print("\n=== Detección de outliers (IQR y Z-score) ===")
    print(tabla.to_string(index=False))


if __name__ == "__main__":
    main()