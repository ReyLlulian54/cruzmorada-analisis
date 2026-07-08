"""
01_load_and_partition.py

Carga el CSV de ventas de Cruz Morada (ventas_completas.csv) usando Dask
(lectura por bloques/paralela), y lo particiona en un "data lake" de
archivos Parquet organizados por LOCAL.

Incluye un benchmark opcional que compara el tiempo de lectura serial
(Pandas, un solo core) contra el tiempo de lectura paralela (Dask,
múltiples cores) sobre una muestra, para demostrar la ganancia real
de paralelismo.

Uso:
    python src/01_load_and_partition.py --input data/ventas_completas.csv --output data/lake
    python src/01_load_and_partition.py --input data/ventas_completas.csv --sample-benchmark

Variables de entorno:
    CPYD_SEED   Semilla fija para reproducibilidad (default: 42)
"""

import argparse
import os
import time
from pathlib import Path

import pandas as pd
import dask.dataframe as dd
from dask.diagnostics import ProgressBar

# ---------------------------------------------------------------------------
# Esquema de columnas según especificación del enunciado.
# Se fuerzan los tipos explícitamente para que Dask no tenga que inferirlos
# leyendo el archivo completo dos veces (más rápido y más seguro con datos
# mixtos, ej. BOLETA que puede traer valores no puramente numéricos).
# ---------------------------------------------------------------------------
COLUMN_DTYPES = {
    "FECHA": "object",              # se parsea a datetime en el script de limpieza
    "CANAL": "object",
    "SKU": "int64",
    "PRODUCTO": "object",
    "UNIDADES": "int64",
    "PORCENTAJE DESCUENTO": "float64",
    "MONTO APLICADO": "float64",
    "BOLETA": "object",
    "LOCAL": "int64",
    "CODIGO CLIENTE": "object",
    "RUN CLIENTE": "object",
    "NOMBRES": "object",
    "APELLIDOS": "object",
    "FECHA NACIMIENTO": "object",
    "GENERO": "Int64",              # Int64 (nullable) por si hay géneros faltantes
}

RENAME_MAP = {
    "FECHA": "fecha",
    "CANAL": "canal",
    "SKU": "sku",
    "PRODUCTO": "producto",
    "UNIDADES": "unidades",
    "PORCENTAJE DESCUENTO": "pct_descuento",
    "MONTO APLICADO": "monto_aplicado",
    "BOLETA": "boleta",
    "LOCAL": "local",
    "CODIGO CLIENTE": "codigo_cliente",
    "RUN CLIENTE": "run_cliente",
    "NOMBRES": "nombres",
    "APELLIDOS": "apellidos",
    "FECHA NACIMIENTO": "fecha_nacimiento",
    "GENERO": "genero",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Carga y particiona ventas_completas.csv en un data lake Parquet."
    )
    parser.add_argument("--input", required=True, help="Ruta al CSV de entrada")
    parser.add_argument(
        "--output", default="data/lake", help="Directorio de salida (data lake Parquet)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("CPYD_SEED", 42)),
        help="Semilla fija (por defecto lee CPYD_SEED del entorno)",
    )
    parser.add_argument(
        "--blocksize",
        default="64MB",
        help="Tamaño de bloque para la lectura paralela de Dask (default: 64MB)",
    )
    parser.add_argument(
        "--sep",
        default=";",
        help="Separador de campos del CSV (default: ';', confirmado con debug_lines.py)",
    )
    parser.add_argument(
        "--sample-benchmark",
        action="store_true",
        help="Corre un benchmark serial (Pandas) vs paralelo (Dask) antes de procesar",
    )
    parser.add_argument(
        "--benchmark-rows",
        type=int,
        default=300_000,
        help="Filas a usar en el benchmark (default: 300,000)",
    )
    return parser.parse_args()


def benchmark_serial_vs_parallel(input_path: str, nrows: int, sep: str) -> dict:
    """Compara lectura serial (Pandas, single-core) vs paralela (Dask, multi-core)."""
    print(f"\n--- Benchmark serial vs paralelo (muestra de {nrows:,} filas) ---")

    t0 = time.time()
    df_pd = pd.read_csv(
        input_path, nrows=nrows, dtype=COLUMN_DTYPES, encoding="utf-8",
        sep=sep, quotechar='"',
    )
    t_serial = time.time() - t0
    print(f"Pandas (serial, 1 core):   {t_serial:.2f}s  -> {len(df_pd):,} filas")

    t0 = time.time()
    ddf = dd.read_csv(
        input_path, dtype=COLUMN_DTYPES, blocksize="16MB", encoding="utf-8",
        sep=sep, quotechar='"',
    )
    df_dask = ddf.head(nrows, npartitions=-1, compute=True)
    t_parallel = time.time() - t0
    print(
        f"Dask (paralelo, {ddf.npartitions} particiones): "
        f"{t_parallel:.2f}s  -> {len(df_dask):,} filas"
    )

    speedup = t_serial / t_parallel if t_parallel > 0 else float("nan")
    print(f"Speedup aproximado: {speedup:.2f}x\n")
    return {"serial_s": t_serial, "parallel_s": t_parallel, "speedup": speedup}


def main():
    args = parse_args()
    os.environ["CPYD_SEED"] = str(args.seed)
    print(f"[INFO] Semilla fija (CPYD_SEED): {args.seed}")

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de entrada: {input_path}")

    if args.sample_benchmark:
        benchmark_serial_vs_parallel(str(input_path), args.benchmark_rows, args.sep)

    print(f"[INFO] Cargando '{input_path}' con Dask (blocksize={args.blocksize})...")
    t0 = time.time()

    ddf = dd.read_csv(
        str(input_path),
        dtype=COLUMN_DTYPES,
        blocksize=args.blocksize,
        encoding="utf-8",
        sep=args.sep,
        quotechar='"',
    )
    ddf = ddf.rename(columns=RENAME_MAP)
    print(f"[INFO] Particiones de lectura (por bloque): {ddf.npartitions}")

    with ProgressBar():
        n_filas = len(ddf)
        n_locales = ddf["local"].nunique().compute()
    print(f"[INFO] Total de filas: {n_filas:,} | Locales únicos: {n_locales}")

    # Reparticionamos explícitamente por LOCAL: esta es la partición lógica
    # de negocio que pide el enunciado para el procesamiento paralelo
    # posterior (limpieza, cálculo de estadísticos por sucursal, etc.)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Escribiendo Parquet particionado por 'local' en '{output_dir}'...")
    with ProgressBar():
        ddf.to_parquet(
            str(output_dir),
            partition_on=["local"],
            engine="pyarrow",
            write_index=False,
        )

    elapsed = time.time() - t0
    print(f"\n[OK] Proceso completo en {elapsed:.2f}s")
    print(f"[OK] Data lake creado en: {output_dir.resolve()}")


if __name__ == "__main__":
    main()