"""

Modelo de regresión (Opción A del enunciado).

    log(monto_aplicado) ~ canal + unidades_producto_boleta + pct_descuento

No se incluye LOCAL: tiene 792 niveles y está casi perfectamente
confundido con CANAL. Incluir ambas variables generaría
multicolinealidad severa sin aportar información nueva.

Se modela log(monto_aplicado) en vez de monto_aplicado directo (o una
versión "winsorizada"/capada), dada su asimetría extrema, el log se ve mucho más cercano a una distribución normal,
y no descarta información real de las transacciones más caras (que no
necesariamente son errores, podrían ser medicamentos de alto costo).

No se usan las columnas monto_aplicado_z / edad_z
pre-calculadas con estadísticos globales; todas las features se
construyen aquí de las columnas crudas, y el split train/test se hace
ANTES de cualquier ajuste.

Diagnósticos: normalidad de residuos (Shapiro-Wilk + Lilliefors sobre
muestra — NO KS estándar, que da p-valores inválidos cuando los
parámetros se estiman de la misma muestra), homocedasticidad
(Breusch-Pagan), VIF sobre TODAS las variables del modelo (numéricas
y dummies de canal).

Métricas en TEST: RMSE y MAE, en escala log y en escala original.

Uso:
    python src/10_regression_model.py --lake data/lake_clean
"""
import argparse
import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import lilliefors, het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


def parse_args():
    parser = argparse.ArgumentParser(description="Modelo de regresión OLS para log(monto_aplicado)")
    parser.add_argument("--lake", default="data/lake_clean", help="Data lake limpio (Parquet)")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument(
        "--seed", type=int, default=int(os.environ.get("CPYD_SEED", 42)),
        help="Semilla fija (por defecto lee CPYD_SEED del entorno)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    print(f"[INFO] Semilla fija (CPYD_SEED): {args.seed}")

    glob_path = f"{args.lake}/**/*.parquet"
    con = duckdb.connect()
    df = con.sql(
        f"SELECT canal, monto_aplicado, unidades_producto_boleta, pct_descuento "
        f"FROM '{glob_path}'"
    ).df()
    print(f"[INFO] Filas cargadas: {len(df):,}")

    df["log_monto"] = np.log(df["monto_aplicado"])

    # Dummies de canal, referencia = POS (el canal más frecuente)
    canal_dummies = pd.get_dummies(df["canal"], prefix="canal", drop_first=False)
    canal_dummies = canal_dummies.drop(columns=["canal_POS"])
    X = pd.concat(
        [df[["unidades_producto_boleta", "pct_descuento"]], canal_dummies], axis=1
    ).astype(float)
    y = df["log_monto"]

    # --- Split train/test 70/30 (ANTES de ajustar cualquier cosa) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )
    print(f"[INFO] Train: {len(X_train):,} filas | Test: {len(X_test):,} filas")

    # --- Ajustar OLS sobre TRAIN ---
    X_train_const = sm.add_constant(X_train)
    modelo = sm.OLS(y_train, X_train_const).fit()
    print("\n=== Resumen del modelo (ajustado sobre TRAIN) ===")
    print(modelo.summary())

    # --- Interpretación aproximada (efecto % sobre monto_aplicado) ---
    print("\n=== Interpretación aproximada de coeficientes (efecto % en monto_aplicado) ===")
    interpretacion = pd.DataFrame(
        {
            "variable": modelo.params.index,
            "coef_log": modelo.params.values,
            "efecto_%_aprox": (np.exp(modelo.params.values) - 1) * 100,
        }
    )
    print(interpretacion.to_string(index=False))

    # --- VIF sobre TODAS las variables del modelo (numéricas + dummies) ---
    print("\n=== VIF (Variance Inflation Factor) — todas las variables del modelo ===")
    vif_data = pd.DataFrame()
    vif_data["variable"] = X_train_const.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X_train_const.values, i)
        for i in range(X_train_const.shape[1])
    ]
    print(vif_data.to_string(index=False))

    # --- Diagnósticos de residuos ---
    residuos = modelo.resid
    muestra_idx = residuos.sample(n=min(5000, len(residuos)), random_state=args.seed).index
    residuos_muestra = residuos.loc[muestra_idx]

    shapiro_stat, shapiro_p = stats.shapiro(residuos_muestra)
    lillie_stat, lillie_p = lilliefors(residuos_muestra, dist="norm")
    print(f"\n=== Normalidad de residuos (muestra n={len(residuos_muestra)}) ===")
    print(f"Shapiro-Wilk: stat={shapiro_stat:.4f}, p={shapiro_p:.4g}")
    print(f"Lilliefors:  stat={lillie_stat:.4f}, p={lillie_p:.4g}")

    bp_stat, bp_p, _, _ = het_breuschpagan(residuos, X_train_const)
    print("\n=== Homocedasticidad (Breusch-Pagan) ===")
    print(f"LM stat={bp_stat:.4f}, p={bp_p:.4g}")

    # --- Gráfico de residuos vs valores ajustados (diagnóstico visual) ---
    output_dir = Path("reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    muestra_plot = np.random.choice(len(residuos), size=min(20000, len(residuos)), replace=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        modelo.fittedvalues.values[muestra_plot],
        residuos.values[muestra_plot],
        alpha=0.15, s=8,
    )
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel("Valores ajustados (log_monto)")
    ax.set_ylabel("Residuos")
    ax.set_title("Residuos vs valores ajustados (muestra de 20,000 puntos)")
    fig.tight_layout()
    fig.savefig(output_dir / "residuos_vs_ajustados.png", dpi=150)
    plt.close(fig)
    print(f"[OK] Gráfico guardado en {output_dir / 'residuos_vs_ajustados.png'}")

    # --- Evaluación en TEST ---
    X_test_const = sm.add_constant(X_test, has_constant="add")
    X_test_const = X_test_const[X_train_const.columns]
    y_pred_log = modelo.predict(X_test_const)

    rmse_log = np.sqrt(mean_squared_error(y_test, y_pred_log))
    mae_log = mean_absolute_error(y_test, y_pred_log)
    print("\n=== Métricas en TEST (escala log) ===")
    print(f"RMSE={rmse_log:.4f}, MAE={mae_log:.4f}")

    y_test_orig = np.exp(y_test)
    y_pred_orig = np.exp(y_pred_log)
    rmse_orig = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
    mae_orig = mean_absolute_error(y_test_orig, y_pred_orig)
    print("\n=== Métricas en TEST (escala original, $) ===")
    print(f"RMSE={rmse_orig:,.2f}, MAE={mae_orig:,.2f}")
    print(
        "[NOTA] Al des-transformar exp(log(y)) se introduce un sesgo leve y "
        "conocido (desigualdad de Jensen); RMSE/MAE en escala original son "
        "aproximados, no exactos."
    )

    print(f"\n[INFO] R² (train) = {modelo.rsquared:.4f} | R² ajustado = {modelo.rsquared_adj:.4f}")


if __name__ == "__main__":
    main()