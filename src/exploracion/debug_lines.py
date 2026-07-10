"""

Inspecciona el CSV crudo como texto plano (sin pandas) para diagnosticar
problemas de parseo: comas sin escapar dentro de campos de texto,
filas con más o menos columnas de las esperadas, problemas de encoding.

Uso:
    # Ver líneas puntuales (ej. el header y las primeras filas)
    python src/debug_lines.py --input data/ventas_completas.csv --start 1 --end 5

    # Ver alrededor de una línea con error
    python src/debug_lines.py --input data/ventas_completas.csv --start 124 --end 132

    # Escanear el archivo completo y contar cuántos "campos" tiene cada línea
    python src/debug_lines.py --input data/ventas_completas.csv --full-scan
"""

import argparse
from collections import Counter


def show_lines(path: str, start: int, end: int) -> None:
    with open(path, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, start=1):
            if i < start:
                continue
            if i > end:
                break
            n_fields = len(line.rstrip("\n").rstrip("\r").split(","))
            print(f"Línea {i} ({n_fields} campos): {line!r}")


def full_scan(path: str) -> None:
    """Cuenta cuántas líneas tienen cada cantidad de 'campos' separados por coma.
    Si el archivo es consistente, casi el 100% de las líneas deberían tener
    la misma cantidad (15, según el enunciado). Cualquier otro número indica
    filas con comas sin escapar dentro de un campo de texto."""
    counts = Counter()
    total = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            n_fields = len(line.rstrip("\n").rstrip("\r").split(","))
            counts[n_fields] += 1
            total += 1

    print(f"Total de líneas leídas: {total:,}\n")
    print("Distribución de cantidad de campos por línea:")
    for n_fields, count in sorted(counts.items()):
        pct = 100 * count / total
        print(f"  {n_fields:3d} campos -> {count:,} líneas ({pct:.4f}%)")


def main():
    parser = argparse.ArgumentParser(description="Diagnóstico de CSV crudo")
    parser.add_argument("--input", required=True, help="Ruta al CSV")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=10)
    parser.add_argument(
        "--full-scan",
        action="store_true",
        help="Escanea el archivo completo y cuenta campos por línea",
    )
    args = parser.parse_args()

    if args.full_scan:
        full_scan(args.input)
    else:
        show_lines(args.input, args.start, args.end)


if __name__ == "__main__":
    main()