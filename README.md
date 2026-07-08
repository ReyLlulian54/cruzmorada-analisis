# Cruz Morada — Análisis Estadístico de Ventas (Computación Paralela y Distribuida)

## Setup

```bash
python -m venv venv
source venv/bin/activate   # En Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
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

1. `src/01_load_and_partition.py` — Carga paralela del CSV + partición a Parquet por LOCAL.
   ```bash
   python src/01_load_and_partition.py --input data/ventas_completas.csv --output data/lake --sample-benchmark
   ```
2. `src/02_cleaning_missing_outliers.py` — Valores faltantes, outliers, variables derivadas. *(pendiente)*
3. `src/03_eda.py` — Estadística descriptiva, visualizaciones, tests de asociación. *(pendiente)*
4. `src/04_inference_modeling.py` — Hipótesis, regresión, validación train/test. *(pendiente)*

## Notas

- Todos los scripts se ejecutan por línea de comandos (`argparse`), tal como exige el enunciado.
- Verificar antes de avanzar: valores reales de la columna `CANAL` (confirmar si existen
  categorías además de `POS`, ya que una hipótesis del enunciado compara APP vs WEB).
