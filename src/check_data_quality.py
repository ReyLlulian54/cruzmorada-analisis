"""
check_data_quality.py

Chequeo profundo de calidad de datos sobre el data lake (data/lake/),
más allá de nulos explícitos: busca valores "centinela" (missing
disfrazado), rangos imposibles y duplicados exactos.

Uso:
    python src/check_data_quality.py --lake data/lake
"""
import argparse
import duckdb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lake", default="data/lake", help="Carpeta del data lake Parquet")
    args = parser.parse_args()

    glob_path = f"{args.lake}/**/*.parquet"
    con = duckdb.connect()

    print("=== GENERO: valores únicos y conteos ===")
    print(con.sql(f"SELECT genero, COUNT(*) AS n FROM '{glob_path}' GROUP BY genero ORDER BY genero").df())

    print("\n=== PCT_DESCUENTO: rango y valores fuera de [0,1] ===")
    print(
        con.sql(
            f"""
            SELECT
                MIN(pct_descuento) AS min_val,
                MAX(pct_descuento) AS max_val,
                COUNT(DISTINCT pct_descuento) AS valores_distintos,
                SUM(CASE WHEN pct_descuento < 0 THEN 1 ELSE 0 END) AS negativos,
                SUM(CASE WHEN pct_descuento > 1 THEN 1 ELSE 0 END) AS mayores_a_1,
                SUM(CASE WHEN pct_descuento = 0 THEN 1 ELSE 0 END) AS iguales_a_0
            FROM '{glob_path}'
            """
        ).df()
    )

    print("\n=== FECHA_NACIMIENTO: fechas no parseables ===")
    print(
        con.sql(
            f"""
            SELECT
                COUNT(*) AS total_filas,
                SUM(CASE WHEN TRY_CAST(fecha_nacimiento AS DATE) IS NULL THEN 1 ELSE 0 END) AS no_parseables,
                SUM(CASE WHEN fecha_nacimiento = '' THEN 1 ELSE 0 END) AS vacios_string
            FROM '{glob_path}'
            """
        ).df()
    )

    print("\n=== EDAD implícita (fecha transacción - fecha nacimiento): rango ===")
    print(
        con.sql(
            f"""
            SELECT
                MIN(DATE_DIFF('year', CAST(fecha_nacimiento AS DATE), CAST(fecha AS DATE))) AS edad_min,
                MAX(DATE_DIFF('year', CAST(fecha_nacimiento AS DATE), CAST(fecha AS DATE))) AS edad_max,
                SUM(CASE WHEN DATE_DIFF('year', CAST(fecha_nacimiento AS DATE), CAST(fecha AS DATE)) < 0 THEN 1 ELSE 0 END) AS edad_negativa,
                SUM(CASE WHEN DATE_DIFF('year', CAST(fecha_nacimiento AS DATE), CAST(fecha AS DATE)) > 110 THEN 1 ELSE 0 END) AS edad_mayor_110
            FROM '{glob_path}'
            WHERE TRY_CAST(fecha_nacimiento AS DATE) IS NOT NULL
            """
        ).df()
    )

    print("\n=== Cadenas vacías en columnas de texto clave ===")
    print(
        con.sql(
            f"""
            SELECT
                SUM(CASE WHEN run_cliente = '' THEN 1 ELSE 0 END)    AS vacios_run,
                SUM(CASE WHEN boleta = '' THEN 1 ELSE 0 END)         AS vacios_boleta,
                SUM(CASE WHEN nombres = '' THEN 1 ELSE 0 END)        AS vacios_nombres,
                SUM(CASE WHEN codigo_cliente = '' THEN 1 ELSE 0 END) AS vacios_codigo_cliente
            FROM '{glob_path}'
            """
        ).df()
    )

    print("\n=== Duplicados exactos (misma fila completa repetida) ===")
    print(
        con.sql(
            f"""
            SELECT COUNT(*) AS filas_duplicadas FROM (
                SELECT *, COUNT(*) OVER (PARTITION BY boleta, sku, codigo_cliente, fecha) AS n
                FROM '{glob_path}'
            ) WHERE n > 1
            """
        ).df()
    )


if __name__ == "__main__":
    main()