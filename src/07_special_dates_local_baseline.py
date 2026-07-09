"""
07_special_dates_local_baseline.py

Refinamiento de la comparación de fechas especiales del Bloque 4.

Comparar una fecha especial contra el promedio de TODO el período
confunde dos efectos: el día de la semana (ej. el día de la madre cae
sábado-domingo, que ya son los días más bajos por defecto) y la fuerte
tendencia de crecimiento de la serie. Este script compara cada fecha
especial contra el promedio del MISMO día de la semana en una ventana
cercana (+/- N semanas), controlando por ambos efectos a la vez.

Uso:
    python src/07_special_dates_local_baseline.py --serie reports/figures/serie_diaria_confiable.csv
"""
import argparse
import pandas as pd

FECHAS_ESPECIALES = {
    "Dia de la madre 2024": ["2024-05-11", "2024-05-12"],
    "Fiestas Patrias 2024": ["2024-09-18", "2024-09-19"],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--serie", default="reports/figures/serie_diaria_confiable.csv",
        help="CSV generado por 06_temporal_patterns.py",
    )
    parser.add_argument("--ventana-semanas", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    serie = pd.read_csv(args.serie, parse_dates=["dia"]).set_index("dia")

    ventana_dias = args.ventana_semanas * 7

    print(
        f"=== Comparación controlada por día de la semana "
        f"(ventana ±{args.ventana_semanas} semanas) ===\n"
    )
    for nombre, fechas in FECHAS_ESPECIALES.items():
        for fecha_str in fechas:
            fecha = pd.Timestamp(fecha_str)
            dia_semana = fecha.day_name()
            valor_real = serie.loc[fecha, "monto_total"]

            ventana = serie.loc[
                fecha - pd.Timedelta(days=ventana_dias): fecha + pd.Timedelta(days=ventana_dias)
            ]
            baseline = ventana[(ventana.index.day_name() == dia_semana) & (ventana.index != fecha)]
            promedio_baseline = baseline["monto_total"].mean()
            n_dias_baseline = len(baseline)

            variacion = (valor_real / promedio_baseline - 1) * 100
            print(
                f"{nombre} - {fecha_str} ({dia_semana}): "
                f"real={valor_real:,.0f} | baseline mismo día (n={n_dias_baseline})={promedio_baseline:,.0f} | "
                f"variación={variacion:+.1f}%"
            )
        print()


if __name__ == "__main__":
    main()