"""
check_ranges_and_duplicates.py

Último chequeo antes de escribir el script de limpieza:
- Rangos de UNIDADES y MONTO_APLICADO
- Si los "duplicados" (mismo boleta+sku+cliente+fecha) son filas
  idénticas de verdad, o difieren en monto/descuento (líneas legítimas)
- Ejemplos concretos de las filas con edad imposible y descuento >100%

Uso:
    python src/check_ranges_and_duplicates.py --lake data/lake
"""
import argparse
import duckdb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lake", default="data/lake", help="Carpeta del data lake Parquet")
    args = parser.parse_args()

    glob_path = f"{args.lake}/**/*.parquet"
    con = duckdb.connect()

    print("=== UNIDADES: rango ===")
    print(
        con.sql(
            f"""
            SELECT MIN(unidades) AS min_val, MAX(unidades) AS max_val,
                   SUM(CASE WHEN unidades <= 0 THEN 1 ELSE 0 END) AS no_positivas
            FROM '{glob_path}'
            """
        ).df()
    )

    print("\n=== MONTO_APLICADO: rango ===")
    print(
        con.sql(
            f"""
            SELECT MIN(monto_aplicado) AS min_val, MAX(monto_aplicado) AS max_val,
                   AVG(monto_aplicado) AS promedio,
                   SUM(CASE WHEN monto_aplicado <= 0 THEN 1 ELSE 0 END) AS no_positivo
            FROM '{glob_path}'
            """
        ).df()
    )

    print("\n=== Grupos con mismo boleta+sku+cliente+fecha: ¿son idénticos o difieren? ===")
    print(
        con.sql(
            f"""
            SELECT boleta, sku, codigo_cliente, fecha, COUNT(*) AS n,
                   COUNT(DISTINCT monto_aplicado) AS montos_distintos,
                   COUNT(DISTINCT pct_descuento) AS descuentos_distintos,
                   COUNT(DISTINCT unidades) AS unidades_distintas
            FROM '{glob_path}'
            GROUP BY boleta, sku, codigo_cliente, fecha
            HAVING COUNT(*) > 1
            ORDER BY n DESC
            LIMIT 10
            """
        ).df()
    )

    print("\n=== Duplicados EXACTOS (las 15 columnas iguales) ===")
    print(
        con.sql(
            f"""
            SELECT COUNT(*) AS filas_totales,
                   COUNT(*) - COUNT(DISTINCT CONCAT_WS('|',
                       fecha, canal, CAST(sku AS VARCHAR), producto,
                       CAST(unidades AS VARCHAR), CAST(pct_descuento AS VARCHAR),
                       CAST(monto_aplicado AS VARCHAR), boleta, CAST(local AS VARCHAR),
                       codigo_cliente, run_cliente, nombres, apellidos,
                       fecha_nacimiento, CAST(genero AS VARCHAR)
                   )) AS filas_duplicadas_exactas
            FROM '{glob_path}'
            """
        ).df()
    )

    print("\n=== Ejemplos: filas con PCT_DESCUENTO > 1 ===")
    print(
        con.sql(
            f"SELECT fecha, canal, sku, producto, pct_descuento, monto_aplicado, local "
            f"FROM '{glob_path}' WHERE pct_descuento > 1"
        ).df()
    )

    print("\n=== Ejemplos: 5 filas con la edad implícita MÁS BAJA (negativa) ===")
    print(
        con.sql(
            f"""
            SELECT fecha, fecha_nacimiento,
                   DATE_DIFF('year', CAST(fecha_nacimiento AS DATE), CAST(fecha AS DATE)) AS edad,
                   codigo_cliente
            FROM '{glob_path}'
            ORDER BY edad ASC
            LIMIT 5
            """
        ).df()
    )

    print("\n=== Ejemplos: 5 filas con la edad implícita MÁS ALTA ===")
    print(
        con.sql(
            f"""
            SELECT fecha, fecha_nacimiento,
                   DATE_DIFF('year', CAST(fecha_nacimiento AS DATE), CAST(fecha AS DATE)) AS edad,
                   codigo_cliente
            FROM '{glob_path}'
            ORDER BY edad DESC
            LIMIT 5
            """
        ).df()
    )


if __name__ == "__main__":
    main()