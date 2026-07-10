# Cruz Morada — Análisis Estadístico de Ventas (Computación Paralela y Distribuida)

## Setup

```bash
python -m venv venv
source venv/bin/activate   # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Colocar el archivo `ventas_completas.csv` dentro de `data/` (no se sube a git, revisar `.gitignore`).

## Arquitectura

- **Paralelismo**: Dask (`dask.dataframe`) para carga y transformación por bloques/particiones.
  Se complementa con `multiprocessing`/`concurrent.futures` para pruebas estadísticas
  que requieren repetición (bootstrapping, permutation tests).
- **Almacenamiento ("data lake")**: Parquet particionado por `LOCAL` en `data/lake/`.
- **Consultas analíticas**: DuckDB sobre los archivos Parquet (SQL embebido, sin servidor).
- **Reproducibilidad**: semilla fija leída desde la variable de entorno `CPYD_SEED` (default 42).

## Pipeline (orden de ejecución)

El proyecto cuenta con un orquestador principal (`main.py`) que ejecuta secuencialmente todos los scripts del pipeline y genera un resumen ordenado en consola:

```bash
python main.py
```

También es posible ejecutar scripts de manera individual. Los scripts que componen el flujo de trabajo son:

1. `src/01_load_and_partition.py` — Carga paralela del CSV y partición a Parquet por LOCAL (incluye validación de paralelismo con Dask).
2. `src/02_cleaning_and_features.py` — Limpieza de datos (fechas, variables numéricas) y creación de features derivadas.
3. `src/03_descriptive_stats.py` — Estadística descriptiva automatizada usando DuckDB.
4. `src/04_histograms_normality.py` — Generación de histogramas, boxplots y tests formales de normalidad.
5. `src/05_association_analysis.py` — Análisis de asociación de variables (Correlación de Spearman, Chi-cuadrado, Kruskal-Wallis).
6. `src/06_temporal_patterns.py` — Patrones temporales y descomposición STL (tendencia, estacionalidad, residual).
7. `src/07_special_dates_local_baseline.py` — Comparación de métricas en fechas especiales frente a su baseline histórica.
8. `src/08_hypothesis_testing.py` — Pruebas de hipótesis especificadas en el requerimiento (t-tests, ANOVA).
9. `src/09_own_hypotheses.py` — Comprobación de hipótesis propias (efectos de género, edad y canal de venta).
10. `src/10_regression_model.py` — Modelo de regresión lineal (Ridge) y análisis de residuales.
11. `src/11_outlier_detection.py` — Diagnóstico adicional de outliers mediante rango intercuartil (IQR) y Z-score.

## Notas

- Todos los scripts se ejecutan por línea de comandos (`argparse`), tal como exige el enunciado.
- Verificar antes de avanzar: valores reales de la columna `CANAL` (confirmar si existen
  categorías además de `POS`, ya que una hipótesis del enunciado compara APP vs WEB).
