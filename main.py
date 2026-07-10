"""
main.py — Orquestador del Pipeline de Análisis de Ventas Cruz Morada

Ejecuta secuencialmente los 11 scripts del proyecto, mostrando
un resumen limpio y estructurado del progreso y los resultados
de cada etapa.

Uso:
    python main.py                  (ejecuta todo el pipeline)
    python main.py --desde 3        (ejecuta desde el script 03 en adelante)
    python main.py --solo 10        (ejecuta solo el script 10)

Variables de entorno:
    CPYD_SEED   Semilla fija para reproducibilidad (default: 42)
"""

import argparse
import subprocess
import sys
import time
import os
from dotenv import load_dotenv

load_dotenv()

# Forzar UTF-8 en la consola de Windows para caracteres especiales
if sys.platform == "win32":
    os.system("chcp 65001 > nul 2>&1")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────────────────────
# Definición del pipeline: (número, archivo, descripción corta)
# ─────────────────────────────────────────────────────────────
PIPELINE = [
    (1,  "src/01_load_and_partition.py",          "Carga CSV y partición a Data Lake (Parquet)",
         ["--input", "data/ventas_completas.csv"]),
    (2,  "src/02_cleaning_and_features.py",       "Limpieza de datos y features derivados",
         []),
    (3,  "src/03_descriptive_stats.py",           "Estadística descriptiva (DuckDB)",
         []),
    (4,  "src/04_histograms_normality.py",        "Histogramas y tests de normalidad",
         []),
    (5,  "src/05_association_analysis.py",         "Análisis de asociación (Spearman, Chi², Kruskal-Wallis)",
         []),
    (6,  "src/06_temporal_patterns.py",            "Patrones temporales y descomposición STL",
         []),
    (7,  "src/07_special_dates_local_baseline.py", "Comparación de fechas especiales vs baseline",
         []),
    (8,  "src/08_hypothesis_testing.py",           "Pruebas de hipótesis del enunciado",
         []),
    (9,  "src/09_own_hypotheses.py",               "Hipótesis propias (género, edad, canal)",
         []),
    (10, "src/10_regression_model.py",             "Modelo de regresión Ridge",
         []),
    (11, "src/11_outlier_detection.py",            "Detección avanzada de outliers (IQR y Z-score)",
         []),
]

# ─────────────────────────────────────────────────────────────
# Constantes de formato
# ─────────────────────────────────────────────────────────────
LINE_WIDTH = 76
BOLD    = "\033[1m"
GREEN   = "\033[92m"
RED     = "\033[91m"
CYAN    = "\033[96m"
YELLOW  = "\033[93m"
DIM     = "\033[2m"
MAGENTA = "\033[95m"
BLUE    = "\033[94m"
RESET   = "\033[0m"


def banner():
    """Imprime el banner de inicio del proyecto."""
    print()
    print(f"{CYAN}{'═' * LINE_WIDTH}")
    print(f"  ╔{'═' * (LINE_WIDTH - 4)}╗")
    print(f"  ║{' ' * ((LINE_WIDTH - 4 - 36) // 2)}ANÁLISIS DE VENTAS — CRUZ MORADA{' ' * ((LINE_WIDTH - 4 - 36) // 2 + (LINE_WIDTH - 4 - 36) % 2)}║")
    print(f"  ║{' ' * ((LINE_WIDTH - 4 - 34) // 2)}Computación Paralela y Distribuida{' ' * ((LINE_WIDTH - 4 - 34) // 2 + (LINE_WIDTH - 4 - 34) % 2)}║")
    print(f"  ╚{'═' * (LINE_WIDTH - 4)}╝")
    print(f"{'═' * LINE_WIDTH}{RESET}")
    seed = os.environ.get("CPYD_SEED", "42")
    print(f"  {DIM}Semilla de reproducibilidad (CPYD_SEED): {seed}{RESET}")
    print()


def section_header(num, total, description):
    """Imprime el encabezado de cada script."""
    print(f"{CYAN}{'─' * LINE_WIDTH}{RESET}")
    print(f"  {BOLD}[{num:02d}/{total:02d}]{RESET}  {description}")
    print(f"{CYAN}{'─' * LINE_WIDTH}{RESET}")


def run_script(script_path, extra_args=None):
    """Ejecuta un script Python como subproceso y retorna (éxito, duración, salida)."""
    start = time.time()
    child_env = os.environ.copy()
    child_env["PYTHONUTF8"] = "1"
    cmd = [sys.executable, script_path] + (extra_args or [])
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=child_env,
    )
    elapsed = time.time() - start

    # Combinar stdout y stderr
    output = ""
    if result.stdout:
        output += result.stdout
    if result.stderr:
        # Filtrar warnings comunes de matplotlib/dask que ensucian la salida
        stderr_lines = result.stderr.strip().split("\n")
        relevant_errors = [
            line for line in stderr_lines
            if not any(skip in line for skip in [
                "UserWarning", "FutureWarning", "DeprecationWarning",
                "matplotlib", "Matplotlib", "warnings.warn",
                "from dask", "import dask",
            ])
        ]
        if relevant_errors:
            output += "\n".join(relevant_errors)

    success = result.returncode == 0
    return success, elapsed, output.strip()


def format_duration(seconds):
    """Formatea duración en un string legible."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.1f}s"


def print_output(output):
    """Imprime la salida de un script con indentación limpia."""
    if not output:
        return
    for line in output.split("\n"):
        stripped = line.strip()
        prefix = f"{BLUE}│{RESET}  "
        if stripped.startswith("===") or stripped.startswith("───"):
            print(f"{prefix}{CYAN}{BOLD}{line}{RESET}")
        elif any(kw in stripped.upper() for kw in ["RMSE", "MAE", "R²", "R2", "P-VALUE", "STAT", "CHI2", "VIF"]):
            print(f"{prefix}{YELLOW}{line}{RESET}")
        elif stripped.startswith("[OK]"):
            print(f"{prefix}{GREEN}{line}{RESET}")
        elif stripped.startswith("[INFO]"):
            print(f"{prefix}{DIM}{line}{RESET}")
        elif stripped.startswith("[AVISO]") or stripped.startswith("[NOTA]"):
            print(f"{prefix}{MAGENTA}{line}{RESET}")
        elif stripped.startswith("[ERROR]"):
            print(f"{prefix}{RED}{BOLD}{line}{RESET}")
        else:
            print(f"{prefix}{line}")


def summary_table(results):
    """Imprime la tabla resumen final de ejecución."""
    print()
    print(f"{CYAN}╔{'═' * (LINE_WIDTH - 2)}╗{RESET}")
    print(f"{CYAN}║{RESET} {BOLD}RESUMEN DE EJECUCIÓN{RESET}{' ' * (LINE_WIDTH - 24)}{CYAN}║{RESET}")
    print(f"{CYAN}╠{'═' * 5}╦{'═' * 46}╦{'═' * 10}╦{'═' * 10}╣{RESET}")
    print(f"{CYAN}║{RESET} {BOLD}{'#':<3}{RESET} {CYAN}║{RESET} {BOLD}{'Script':<44}{RESET} {CYAN}║{RESET} {BOLD}{'Tiempo':>8}{RESET} {CYAN}║{RESET} {BOLD}{'Estado':>8}{RESET} {CYAN}║{RESET}")
    print(f"{CYAN}╠{'═' * 5}╬{'═' * 46}╬{'═' * 10}╬{'═' * 10}╣{RESET}")

    total_time = 0
    all_ok = True
    for num, name, success, elapsed in results:
        status = f"{GREEN}  ✓ OK  {RESET}" if success else f"{RED} ✗ FALLO{RESET}"
        if not success:
            all_ok = False
        total_time += elapsed
        short_name = name.replace("src/", "")
        print(f"{CYAN}║{RESET} {num:02d}  {CYAN}║{RESET} {short_name:<44} {CYAN}║{RESET} {format_duration(elapsed):>8} {CYAN}║{RESET} {status} {CYAN}║{RESET}")

    print(f"{CYAN}╠{'═' * 5}╩{'═' * 46}╬{'═' * 10}╬{'═' * 10}╣{RESET}")
    print(f"{CYAN}║{RESET} {'TOTAL':<52} {CYAN}║{RESET} {format_duration(total_time):>8} {CYAN}║{RESET} {' ' * 8} {CYAN}║{RESET}")
    print(f"{CYAN}╚{'═' * 53}╩{'═' * 10}╩{'═' * 10}╝{RESET}")
    print()

    if all_ok:
        print(f"  {GREEN}{BOLD} Pipeline completo ejecutado exitosamente.{RESET}")
    else:
        failed = [f"{n:02d}" for n, _, s, _ in results if not s]
        print(f"  {RED}{BOLD} Errores en los scripts: {', '.join(failed)}{RESET}")

    print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Orquestador del pipeline de análisis Cruz Morada"
    )
    parser.add_argument(
        "--desde", type=int, default=1, metavar="N",
        help="Ejecutar desde el script N en adelante (default: 1)",
    )
    parser.add_argument(
        "--solo", type=int, default=None, metavar="N",
        help="Ejecutar solo el script N",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    banner()

    # Filtrar scripts según argumentos
    if args.solo is not None:
        scripts = [(n, f, d, a) for n, f, d, a in PIPELINE if n == args.solo]
        if not scripts:
            print(f"{RED}  Error: No existe el script {args.solo}.{RESET}")
            sys.exit(1)
    else:
        scripts = [(n, f, d, a) for n, f, d, a in PIPELINE if n >= args.desde]

    total = len(scripts)
    results = []

    for i, (num, script, desc, extra_args) in enumerate(scripts, 1):
        section_header(i, total, desc)
        success, elapsed, output = run_script(script, extra_args)
        print_output(output)
        
        tag = f"{GREEN}{BOLD}✓ OK{RESET}" if success else f"{RED}{BOLD}✗ FALLO{RESET}"
        print(f"{BLUE}╰{'─' * (LINE_WIDTH - 2)}╯{RESET}")
        print(f"   {DIM} Duración: {format_duration(elapsed)} — {tag}")
        print()

        results.append((num, script, success, elapsed))

        # Si un script falla, preguntar si continuar
        if not success:
            print(f"  {RED}{BOLD} El script {num:02d} falló. Continuando con el siguiente...{RESET}")
            print()

    # Tabla resumen
    summary_table(results)


if __name__ == "__main__":
    main()
