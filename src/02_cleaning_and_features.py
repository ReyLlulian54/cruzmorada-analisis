"""

Limpieza de datos y creación de variables derivadas.

Arquitectura híbrida: la lectura y la escritura
final se hacen con Dask (paralelo, particionado), pero las operaciones
de groupby + join (que requieren shuffle entre particiones) se hacen
materializando a Pandas, ya que el dataset (634MB, 3.24M filas) cabe
cómodamente en memoria. Se probó la alternativa 100% Dask y el shuffle
sobre datos particionados por LOCAL (792 particiones, no relacionado
con las claves de join codigo_cliente/boleta+sku) tardaba más de 20
minutos sin terminar; materializando a Pandas esto toma segundos.

Reglas de limpieza aplicadas (ver diagnóstico previo):
  1. PCT_DESCUENTO fuera de [0,1] (imposible) -> se elimina la fila.
  2. EDAD implícita fuera de [0,100] años -> NaN, imputada con mediana.
     Cálculo con /365.25 (no //365) para no sesgar por años bisiestos.
  3. Duplicados boleta+sku+cliente+fecha -> se conservan (no son
     duplicados reales, son unidades individuales con descuentos
     distintos dentro de la misma boleta).
  4. UNIDADES constante = 1 -> se reconstruye la cantidad real
     comprada por producto/transacción agrupando por (boleta, sku).
  5. frecuencia_compra_cliente: usa nunique(boleta), NO count(boleta).
     count() contaba líneas de producto (una
     boleta con 10 productos sumaba 10), no visitas/transacciones
     reales del cliente.

Nota sobre estandarización (monto_aplicado_z, edad_z): estas columnas
se calculan con media/std GLOBALES, válido para EDA. El
modelo de regresión no las usa directamente como features;
cualquier estandarización para el modelo se recalcula después del
train/test split, usando solo estadísticos del set de entrenamiento,
para evitar fuga de datos.

Uso:
    python src/02_cleaning_and_features.py --lake data/lake --output data/lake_clean
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask.diagnostics import ProgressBar

EDAD_MIN_VALIDA = 0
EDAD_MAX_VALIDA = 100


def parse_args():
    parser = argparse.ArgumentParser(description="Limpieza y variables derivadas")
    parser.add_argument("--lake", default="data/lake", help="Data lake de entrada (Parquet)")
    parser.add_argument("--output", default="data/lake_clean", help="Data lake de salida limpio")
    parser.add_argument(
        "--seed", type=int, default=int(os.environ.get("CPYD_SEED", 42)),
        help="Semilla fija (por defecto lee CPYD_SEED del entorno)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["CPYD_SEED"] = str(args.seed)
    np.random.seed(args.seed)
    print(f"[INFO] Semilla fija (CPYD_SEED): {args.seed}")

    t0 = time.time()

    # -------------------------------------------------------------
    # Lectura paralela (Dask) + limpieza elemental (no requiere shuffle)
    # -------------------------------------------------------------
    print(f"[INFO] Cargando data lake desde '{args.lake}' con Dask...")
    ddf = dd.read_parquet(args.lake)

    with ProgressBar():
        n_inicial = len(ddf)
    print(f"[INFO] Filas iniciales: {n_inicial:,}")

    ddf = ddf[(ddf["pct_descuento"] >= 0) & (ddf["pct_descuento"] <= 1)]
    ddf["fecha"] = dd.to_datetime(ddf["fecha"], format="%Y-%m-%dT%H:%M:%S")

    # Pre-validar el año de nacimiento COMO TEXTO antes de convertir a
    # datetime: algunos años están corruptos (ej. "9771", "7968", "0960")
    # y quedan fuera del rango representable por datetime64[ns]
    # (~1677-2262), lo que provocaba un OverflowError al calcular.
    # Se descartan a nivel de string antes de intentar construir la
    # fecha, para nunca llegar a generar el valor imposible.
    anio_str = ddf["fecha_nacimiento"].str.slice(0, 4)
    anio_valido = anio_str.astype("float64").between(1900, 2024)
    ddf["fecha_nacimiento"] = ddf["fecha_nacimiento"].mask(~anio_valido)
    ddf["fecha_nacimiento"] = dd.to_datetime(
        ddf["fecha_nacimiento"], format="%Y-%m-%d", errors="coerce"
    )
    ddf["edad"] = (ddf["fecha"] - ddf["fecha_nacimiento"]).dt.days // 365.25

    # -------------------------------------------------------------
    # Materializar a Pandas: el resto de las operaciones (groupby +
    # join, imputación, estandarización) no se benefician de Dask
    # aquí porque el dataset cabe en memoria y las claves de
    # agrupación no coinciden con el particionado por LOCAL.
    # -------------------------------------------------------------
    print("[INFO] Materializando a Pandas (groupby/join no se benefician de Dask acá)...")
    with ProgressBar():
        df = ddf.compute()
    n_tras_descuento = len(df)
    print(
        f"[LIMPIEZA] PCT_DESCUENTO fuera de [0,1]: "
        f"{n_inicial - n_tras_descuento} filas eliminadas"
    )
    print(f"[INFO] Filas materializadas: {n_tras_descuento:,}")

    # --- Regla 2: EDAD implícita fuera de rango -> NaN, imputar con mediana ---
    # Incluye tanto las edades calculadas fuera de [0,100] como las filas
    # cuya fecha_nacimiento ya quedó en NaN por año inválido/no parseable
    # (ambos casos se imputan con la misma mediana).
    n_anio_invalido = int(df["edad"].isna().sum())
    edad_invalida = (df["edad"] < EDAD_MIN_VALIDA) | (df["edad"] > EDAD_MAX_VALIDA)
    n_edad_fuera_rango = int(edad_invalida.sum())
    df.loc[edad_invalida, "edad"] = np.nan
    mediana_edad = float(df["edad"].median())
    df["edad"] = df["edad"].fillna(mediana_edad)
    print(
        f"[LIMPIEZA] EDAD: {n_edad_fuera_rango:,} filas fuera de "
        f"[{EDAD_MIN_VALIDA},{EDAD_MAX_VALIDA}] años + {n_anio_invalido:,} filas con año de "
        f"nacimiento no parseable -> total {n_edad_fuera_rango + n_anio_invalido:,} "
        f"imputadas con mediana ({mediana_edad:.0f} años)"
    )

    # --- Variables derivadas ---
    df["monto_por_unidad"] = df["monto_aplicado"] / df["unidades"]
    print(
        "[NOTA] UNIDADES es constante = 1 en todo el dataset; por lo tanto "
        "monto_por_unidad es idéntico a monto_aplicado. Se documenta este "
        "hallazgo en el informe."
    )

    df["frecuencia_compra_cliente"] = df.groupby("codigo_cliente")["boleta"].transform("nunique")
    df["unidades_producto_boleta"] = df.groupby(["boleta", "sku"])["unidades"].transform("count")

    # --- Estandarización (z-score) ---
    monto_mean, monto_std = df["monto_aplicado"].mean(), df["monto_aplicado"].std()
    edad_mean, edad_std = df["edad"].mean(), df["edad"].std()
    df["monto_aplicado_z"] = (df["monto_aplicado"] - monto_mean) / monto_std
    df["edad_z"] = (df["edad"] - edad_mean) / edad_std
    print(f"[INFO] Estandarización monto_aplicado -> media={monto_mean:.2f}, std={monto_std:.2f}")
    print(f"[INFO] Estandarización edad -> media={edad_mean:.2f}, std={edad_std:.2f}")

    # -------------------------------------------------------------
    # Volver a Dask solo para la escritura paralela particionada
    # -------------------------------------------------------------
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Escribiendo dataset limpio (paralelo, particionado por LOCAL) en '{output_dir}'...")

    ddf_final = dd.from_pandas(df, npartitions=8)
    with ProgressBar():
        ddf_final.to_parquet(
            str(output_dir), partition_on=["local"], engine="pyarrow", write_index=False
        )

    elapsed = time.time() - t0
    print(f"\n[OK] Limpieza completa en {elapsed:.2f}s")
    print(f"[OK] Filas finales: {len(df):,} (de {n_inicial:,} originales)")


if __name__ == "__main__":
    main()