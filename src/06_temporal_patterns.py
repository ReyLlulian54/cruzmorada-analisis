"""

Patrones temporales.

La serie diaria completa tiene 0 transacciones
entre 2023-11-10 y 2024-04-06 (149 días, solo 8 transacciones el
primer día), artefacto de generación del dataset sintético, no un
cierre real de la farmacia. Luego hay un crecimiento muy pronunciado
hasta fines de 2024. Por eso el análisis de series de tiempo se
restringe al período con actividad consistente (>= 2024-04-07); de lo
contrario el bloque de ceros distorsiona la descomposición STL.

Consecuencia: no es posible comparar navidad (2023 cae en el período
sin datos; el dataset termina el 2024-12-02, antes de navidad 2024).
Se documenta esta limitación y se usa Fiestas Patrias (18-19 sep 2024)
como fecha especial, que sí muestra una caída de ventas clara.

Uso:
    python src/06_temporal_patterns.py --lake data/lake_clean --fecha-inicio 2024-04-07
"""
import argparse
from pathlib import Path

import duckdb
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lake", default="data/lake_clean", help="Data lake limpio (Parquet)")
    parser.add_argument("--output", default="reports/figures", help="Carpeta de salida")
    parser.add_argument(
        "--fecha-inicio", default="2024-04-07",
        help="Inicio del período confiable (antes de esto la actividad es ~0)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    glob_path = f"{args.lake}/**/*.parquet"
    con = duckdb.connect()

    print("[INFO] Agregando ventas diarias (serie completa)...")
    serie_completa = con.sql(
        f"""
        SELECT CAST(fecha AS DATE) AS dia,
               COUNT(*) AS n_transacciones,
               SUM(monto_aplicado) AS monto_total
        FROM '{glob_path}'
        GROUP BY dia ORDER BY dia
        """
    ).df()
    serie_completa["dia"] = pd.to_datetime(serie_completa["dia"])
    serie_completa = serie_completa.set_index("dia").asfreq("D")
    serie_completa["n_transacciones"] = serie_completa["n_transacciones"].fillna(0)
    serie_completa["monto_total"] = serie_completa["monto_total"].fillna(0)

    dias_vacios = int((serie_completa["n_transacciones"] == 0).sum())
    print(
        f"[AVISO] {dias_vacios} de {len(serie_completa)} días tienen 0 transacciones "
        f"(artefacto de generación del dataset, antes de {args.fecha_inicio})"
    )

    # --- Serie restringida al período confiable ---
    serie = serie_completa.loc[args.fecha_inicio:].copy()
    print(
        f"[INFO] Serie confiable: {len(serie)} días, de "
        f"{serie.index.min().date()} a {serie.index.max().date()}"
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    serie["monto_total"].plot(ax=ax)
    ax.set_title(f"Monto total de ventas por día (desde {args.fecha_inicio})")
    ax.set_ylabel("Monto total")
    fig.tight_layout()
    fig.savefig(output_dir / "serie_diaria_monto_confiable.png", dpi=150)
    plt.close(fig)

    # --- STL sobre el período confiable ---
    print("[INFO] Descomposición STL (period=7, período confiable)...")
    stl = STL(serie["monto_total"], period=7, robust=True)
    resultado = stl.fit()
    fig = resultado.plot()
    fig.set_size_inches(12, 8)
    fig.suptitle("Descomposición STL (período confiable, desde abril 2024)")
    fig.tight_layout()
    fig.savefig(output_dir / "stl_decomposicion_confiable.png", dpi=150)
    plt.close(fig)

    # --- ACF / PACF ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_acf(serie["monto_total"], lags=30, ax=axes[0])
    axes[0].set_title("ACF (período confiable)")
    plot_pacf(serie["monto_total"], lags=30, ax=axes[1])
    axes[1].set_title("PACF (período confiable)")
    fig.tight_layout()
    fig.savefig(output_dir / "acf_pacf_confiable.png", dpi=150)
    plt.close(fig)

    # --- Patrón por día de la semana (explica el ciclo de 7 días) ---
    print("\n=== Promedio de monto_total por día de la semana ===")
    dias_es = {
        "Monday": "Lunes", "Tuesday": "Martes", "Wednesday": "Miércoles",
        "Thursday": "Jueves", "Friday": "Viernes", "Saturday": "Sábado", "Sunday": "Domingo",
    }
    serie["dia_semana"] = serie.index.day_name().map(dias_es)
    orden = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    promedio_dow = serie.groupby("dia_semana")["monto_total"].mean().reindex(orden)
    print(promedio_dow)

    fig, ax = plt.subplots(figsize=(8, 5))
    promedio_dow.plot(kind="bar", ax=ax)
    ax.set_title("Promedio de monto_total por día de la semana")
    ax.set_ylabel("Monto total promedio")
    fig.tight_layout()
    fig.savefig(output_dir / "promedio_por_dia_semana.png", dpi=150)
    plt.close(fig)

    # --- Fechas especiales (dentro del período confiable) ---
    print("\n=== Comparación con fechas especiales (período confiable) ===")
    print(
        "[NOTA] No se compara navidad: navidad 2023 cae en el período sin datos, "
        "y el dataset termina el 2024-12-02, antes de navidad 2024."
    )
    fechas_especiales = {
        "Dia de la madre 2024 (11-12 mayo)": ("2024-05-11", "2024-05-12"),
        "Fiestas Patrias 2024 (18-19 septiembre)": ("2024-09-18", "2024-09-19"),
    }
    promedio_general = serie["monto_total"].mean()
    print(f"Promedio diario (período confiable): {promedio_general:,.2f}")
    for nombre, (ini, fin) in fechas_especiales.items():
        sub = serie.loc[ini:fin, "monto_total"]
        variacion = (sub.mean() / promedio_general - 1) * 100
        print(f"{nombre}: promedio={sub.mean():,.2f} ({variacion:+.1f}% vs. promedio general)")

    serie.to_csv(output_dir / "serie_diaria_confiable.csv")
    print(f"\n[OK] Figuras y resultados guardados en '{output_dir}'")


if __name__ == "__main__":
    main()