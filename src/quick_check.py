"""
quick_check.py

Chequeos rápidos sobre el data lake ya particionado (data/lake/), usando
DuckDB para consultar directamente los archivos Parquet con SQL (sin
cargar todo a memoria).

Responde preguntas clave antes de avanzar a limpieza/transformación:
- ¿Qué valores tiene CANAL? (necesario para la hipótesis APP vs WEB)
- ¿Cuántos nulos hay en las columnas que pide tratar el enunciado?
- ¿Qué rango de fechas cubre el dataset?

Uso:
    python src/quick_check.py --lake data/lake
"""
import argparse
import duckdb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lake", default="data/lake", help="Carpeta del data lake Parquet")
    args = parser.parse_args()

    glob_path = f"{args.lake}/**/*.parquet"
    con = duckdb.connect()

    print("=== Valores únicos de 'canal' ===")
    print(
        con.sql(
            f"SELECT canal, COUNT(*) AS n FROM '{glob_path}' "
            f"GROUP BY canal ORDER BY n DESC"
        ).df()
    )

    print("\n=== Rango de fechas ===")
    print(
        con.sql(f"SELECT MIN(fecha) AS min_fecha, MAX(fecha) AS max_fecha FROM '{glob_path}'").df()
    )

    print("\n=== Nulos en columnas clave (y total de filas) ===")
    print(
        con.sql(
            f"""
            SELECT
                SUM(CASE WHEN pct_descuento IS NULL THEN 1 ELSE 0 END)   AS nulos_pct_descuento,
                SUM(CASE WHEN fecha_nacimiento IS NULL THEN 1 ELSE 0 END) AS nulos_fecha_nacimiento,
                SUM(CASE WHEN genero IS NULL THEN 1 ELSE 0 END)          AS nulos_genero,
                SUM(CASE WHEN monto_aplicado IS NULL THEN 1 ELSE 0 END) AS nulos_monto,
                COUNT(*) AS total_filas
            FROM '{glob_path}'
            """
        ).df()
    )

    print("\n=== Top 5 locales por volumen de transacciones ===")
    print(
        con.sql(
            f"SELECT local, COUNT(*) AS n FROM '{glob_path}' "
            f"GROUP BY local ORDER BY n DESC LIMIT 5"
        ).df()
    )


if __name__ == "__main__":
    main()