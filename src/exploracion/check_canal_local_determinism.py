"""
check_canal_local_determinism.py

Verifica si LOCAL predice CANAL de forma casi determinística (hipótesis
surgida del chi-cuadrado del Bloque 3, donde chi2 dio exactamente = N).

Para cada LOCAL, calcula qué % de sus transacciones caen en su canal
más frecuente. Si la mayoría de los locales están cerca del 100%,
confirma que LOCAL y CANAL están altamente asociados (relevante para
evitar multicolinealidad en el modelo de regresión del Día 3).

Uso:
    python src/check_canal_local_determinism.py --lake data/lake_clean
"""
import argparse
import duckdb
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lake", default="data/lake_clean")
    args = parser.parse_args()

    glob_path = f"{args.lake}/**/*.parquet"
    con = duckdb.connect()
    df = con.sql(f"SELECT canal, local FROM '{glob_path}'").df()

    tabla = df.groupby(["local", "canal"]).size().unstack(fill_value=0)
    pct_dominante = tabla.max(axis=1) / tabla.sum(axis=1)

    print("=== % de concentración en el canal dominante, por LOCAL ===")
    print(pct_dominante.describe())

    print(f"\nLocales 100% de un solo canal: {(pct_dominante == 1.0).sum()} de {len(pct_dominante)} "
          f"({(pct_dominante == 1.0).mean() * 100:.1f}%)")
    print(f"Locales con >=95% de un solo canal: {(pct_dominante >= 0.95).mean() * 100:.1f}%")

    print("\n=== Canal dominante por LOCAL: distribución ===")
    canal_dominante = tabla.idxmax(axis=1)
    print(canal_dominante.value_counts())

    print("\n=== Ejemplo: locales que SÍ mezclan canales (si existen) ===")
    mixtos = pct_dominante[pct_dominante < 0.95].sort_values()
    print(f"Cantidad de locales mixtos: {len(mixtos)}")
    if len(mixtos) > 0:
        print(tabla.loc[mixtos.index[:10]])


if __name__ == "__main__":
    main()