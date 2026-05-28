"""
Módulo: model_monitoring.py
Propósito: Monitoreo de data drift para el modelo de riesgo crediticio.

Este script compara la distribución de un conjunto de datos de referencia
(los datos de entrenamiento) con la de datos nuevos, para detectar
desviaciones estadísticas (data drift) que podrían degradar el desempeño
del modelo en producción.

Técnicas implementadas
-----------------------
1. PSI (Population Stability Index): métrica estándar en riesgo crediticio.
   - PSI < 0.10  : sin drift significativo.
   - 0.10 - 0.25 : drift moderado (vigilar).
   - PSI >= 0.25 : drift severo (considerar reentrenamiento).
2. Kolmogorov-Smirnov (KS): prueba para variables numéricas continuas.
3. Chi-cuadrado: prueba para variables categóricas.

Escenarios de prueba
---------------------
Como el dataset es estático, se evalúan dos escenarios para demostrar el
funcionamiento del detector:
    - Escenario 1 (sin drift): train vs test. No se espera drift relevante.
    - Escenario 2 (con drift inducido): train vs test modificado
      artificialmente. Se espera detectar drift.

Artefactos generados
---------------------
- ``reports/monitoring/psi_report.csv``
- ``reports/monitoring/ks_report.csv``
- ``reports/monitoring/drift_summary.png``
- ``reports/monitoring/distribuciones_drift.png``
- ``artifacts/monitoring/drift_baseline.json``

Uso
---
Desde la raíz del repositorio (requiere haber ejecutado antes
``ft_engineering.py`` para disponer de los datos):

    python mlops_pipeline/src/model_monitoring.py

Autor: Cristian García
Fecha de creación: 2026-05-28
Versión: V1.3.0 (esqueleto inicial)
"""

# =========================================================================
# 1. IMPORTS
# =========================================================================
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ks_2samp, chi2_contingency
from sklearn.model_selection import train_test_split


# =========================================================================
# 2. CONSTANTES Y CONFIGURACIÓN
# =========================================================================

RUTA_RAIZ = Path(__file__).resolve().parents[2]

# Inputs
RUTA_DATASET = RUTA_RAIZ / "Base_de_datos.xlsx"

# Outputs
RUTA_ARTIFACTS = RUTA_RAIZ / "artifacts"
RUTA_MONITORING_ARTIFACTS = RUTA_ARTIFACTS / "monitoring"
RUTA_REPORTS = RUTA_RAIZ / "reports" / "monitoring"

# Columna objetivo (se excluye del análisis de drift de features)
COL_TARGET = "Pago_atiempo"

# Semilla para reproducibilidad
SEMILLA = 42

# Umbrales de interpretación del PSI
PSI_UMBRAL_MODERADO = 0.10
PSI_UMBRAL_SEVERO = 0.25

# Número de bins para el cálculo del PSI en variables numéricas
PSI_N_BINS = 10

# Nivel de significancia para las pruebas estadísticas (KS, Chi-cuadrado)
ALPHA = 0.05

# Proporción del split train/test (debe coincidir con ft_engineering.py)
TEST_SIZE = 0.20

# -----------------------------------------------------------------------------
# Clasificación de columnas para el análisis de drift (sobre datos crudos).
# Se excluyen el target, la fecha y la variable con leakage ('puntaje').
# -----------------------------------------------------------------------------
COLS_NUMERICAS = [
    "edad_cliente", "salario_cliente", "capital_prestado", "cuota_pactada",
    "plazo_meses", "total_otros_prestamos", "saldo_mora", "saldo_total",
    "saldo_principal", "puntaje_datacredito", "cant_creditosvigentes",
    "huella_consulta", "promedio_ingresos_datacredito",
    "creditos_sectorFinanciero", "creditos_sectorReal",
    "creditos_sectorCooperativo",
]

COLS_CATEGORICAS = [
    "tipo_credito", "saldo_mora_codeudor", "tipo_laboral", "tendencia_ingresos",
]

# Columnas excluidas del análisis (target, fecha, leakage)
COLS_EXCLUIDAS = ["Pago_atiempo", "fecha_prestamo", "puntaje"]

# Reglas mínimas de limpieza (deben ser consistentes con ft_engineering.py).
# Solo capping/NaN por dominio; NO se escala ni encodea.
REGLAS_LIMPIEZA = {
    "edad_cliente": {"lo": 18, "hi": 100, "low": "nan", "high": "nan"},
    "salario_cliente": {"lo": 0, "hi": 100_000_000, "low": "cap", "high": "cap"},
    "puntaje_datacredito": {"lo": 1, "hi": 950, "low": "nan", "high": "cap"},
    "total_otros_prestamos": {"lo": 0, "hi": 500_000_000, "low": "cap", "high": "cap"},
}
TENDENCIA_INGRESOS_VALIDAS = ["Creciente", "Estable", "Decreciente"]


# =========================================================================
# 3. CONFIGURACIÓN DE LOGGING
# =========================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =========================================================================
# 4. FUNCIONES DE CÁLCULO DE DRIFT
# =========================================================================

def interpretar_psi(psi: float) -> str:
    """
    Traduce un valor de PSI a su interpretación de negocio.

    Parameters
    ----------
    psi : float
        Valor del Population Stability Index.

    Returns
    -------
    str
        'sin_drift', 'drift_moderado' o 'drift_severo'.
    """
    if psi < PSI_UMBRAL_MODERADO:
        return "sin_drift"
    elif psi < PSI_UMBRAL_SEVERO:
        return "drift_moderado"
    else:
        return "drift_severo"


def calcular_psi(
    referencia: pd.Series,
    nuevos: pd.Series,
    n_bins: int = PSI_N_BINS,
) -> dict:
    """
    Calcula el Population Stability Index (PSI) entre dos distribuciones
    numéricas.

    El PSI mide cuánto se ha desplazado la distribución de una variable
    respecto a una referencia. Los bins se definen por percentiles de la
    distribución de referencia.

    Fórmula:
        PSI = sum( (pct_nuevos - pct_ref) * ln(pct_nuevos / pct_ref) )

    Parameters
    ----------
    referencia : pd.Series
        Valores de la variable en el conjunto de referencia (train).
    nuevos : pd.Series
        Valores de la variable en el conjunto nuevo.
    n_bins : int
        Número de bins (cajones) a usar.

    Returns
    -------
    dict
        - 'psi': valor del índice.
        - 'interpretacion': categoría de drift.
    """
    # Eliminar nulos para el cálculo
    ref = referencia.dropna()
    nue = nuevos.dropna()

    # Definir los bordes de los bins usando percentiles de la referencia.
    # Usamos np.unique para evitar bordes duplicados (variables con muchos
    # valores repetidos).
    percentiles = np.linspace(0, 100, n_bins + 1)
    bordes = np.unique(np.percentile(ref, percentiles))

    # Si la variable tiene muy pocos valores únicos, puede haber menos
    # bordes que bins solicitados; se ajusta automáticamente.
    if len(bordes) < 2:
        # Variable casi constante: no hay drift medible.
        return {"psi": 0.0, "interpretacion": "sin_drift"}

    # Asegurar que los extremos capturen todos los valores
    bordes[0] = -np.inf
    bordes[-1] = np.inf

    # Contar proporción de datos en cada bin
    ref_counts, _ = np.histogram(ref, bins=bordes)
    nue_counts, _ = np.histogram(nue, bins=bordes)

    ref_pct = ref_counts / len(ref)
    nue_pct = nue_counts / len(nue)

    # Sustituir ceros por un valor mínimo para evitar ln(0) o división por 0
    epsilon = 1e-4
    ref_pct = np.where(ref_pct == 0, epsilon, ref_pct)
    nue_pct = np.where(nue_pct == 0, epsilon, nue_pct)

    # Calcular PSI
    psi = np.sum((nue_pct - ref_pct) * np.log(nue_pct / ref_pct))

    return {"psi": float(psi), "interpretacion": interpretar_psi(float(psi))}


def calcular_ks(referencia: pd.Series, nuevos: pd.Series) -> dict:
    """
    Aplica la prueba de Kolmogorov-Smirnov de dos muestras.

    Compara si dos muestras numéricas provienen de la misma distribución.
    Un p-valor < ALPHA indica que las distribuciones difieren
    significativamente (hay drift).

    Parameters
    ----------
    referencia, nuevos : pd.Series
        Valores numéricos a comparar.

    Returns
    -------
    dict
        - 'ks_statistic': estadístico KS (0 a 1; mayor = más diferencia).
        - 'p_value': p-valor de la prueba.
        - 'hay_drift': True si p_value < ALPHA.
    """
    ref = referencia.dropna()
    nue = nuevos.dropna()

    estadistico, p_value = ks_2samp(ref, nue)

    return {
        "ks_statistic": float(estadistico),
        "p_value": float(p_value),
        "hay_drift": bool(p_value < ALPHA),
    }


def calcular_chi2(referencia: pd.Series, nuevos: pd.Series) -> dict:
    """
    Aplica la prueba de Chi-cuadrado de independencia para variables
    categóricas.

    Compara las frecuencias de cada categoría entre los dos conjuntos.
    Un p-valor < ALPHA indica que las distribuciones de categorías
    difieren significativamente (hay drift).

    Parameters
    ----------
    referencia, nuevos : pd.Series
        Valores categóricos a comparar.

    Returns
    -------
    dict
        - 'chi2_statistic': estadístico Chi-cuadrado.
        - 'p_value': p-valor de la prueba.
        - 'hay_drift': True si p_value < ALPHA.
    """
    # Construir tabla de contingencia: filas = categorías, columnas = grupos
    ref_counts = referencia.value_counts()
    nue_counts = nuevos.value_counts()

    # Unificar el conjunto de categorías de ambos grupos
    todas_categorias = sorted(set(ref_counts.index) | set(nue_counts.index),
                              key=lambda x: str(x))

    tabla = pd.DataFrame({
        "referencia": [ref_counts.get(cat, 0) for cat in todas_categorias],
        "nuevos": [nue_counts.get(cat, 0) for cat in todas_categorias],
    }, index=todas_categorias)

    # Eliminar categorías con 0 en ambos grupos (no aportan)
    tabla = tabla[(tabla["referencia"] + tabla["nuevos"]) > 0]

    # Chi-cuadrado requiere al menos 2 categorías
    if len(tabla) < 2:
        return {"chi2_statistic": 0.0, "p_value": 1.0, "hay_drift": False}

    chi2, p_value, _, _ = chi2_contingency(tabla.values)

    return {
        "chi2_statistic": float(chi2),
        "p_value": float(p_value),
        "hay_drift": bool(p_value < ALPHA),
    }


# =========================================================================
# 5. GENERACIÓN DE ESCENARIOS
# =========================================================================

def limpiar_minimo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica una limpieza mínima al dataset, consistente con la lógica de
    ``ft_engineering.py`` pero SIN escalado ni encoding.

    Operaciones:
        - tendencia_ingresos: NaN -> 'DESCONOCIDO', categorías válidas se
          conservan, resto -> 'OTROS'.
        - capping/NaN por dominio para las variables de REGLAS_LIMPIEZA.

    Esta función se re-implementa localmente (en vez de importarla de
    ft_engineering) para evitar acoplamiento entre módulos.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset crudo.

    Returns
    -------
    pd.DataFrame
        Dataset con limpieza mínima aplicada.
    """
    df = df.copy()

    # Limpieza de tendencia_ingresos
    if "tendencia_ingresos" in df.columns:
        def _mapear(v):
            if pd.isna(v):
                return "DESCONOCIDO"
            if v in TENDENCIA_INGRESOS_VALIDAS:
                return v
            return "OTROS"
        df["tendencia_ingresos"] = df["tendencia_ingresos"].apply(_mapear)

    # Capping/NaN por dominio
    for col, reglas in REGLAS_LIMPIEZA.items():
        if col not in df.columns:
            continue
        lo, hi = reglas["lo"], reglas["hi"]
        if reglas["low"] == "nan":
            df.loc[df[col] < lo, col] = np.nan
        else:
            df.loc[df[col] < lo, col] = lo
        if reglas["high"] == "nan":
            df.loc[df[col] > hi, col] = np.nan
        else:
            df.loc[df[col] > hi, col] = hi

    return df


def inducir_drift(df: pd.DataFrame) -> pd.DataFrame:
    """
    Induce drift artificial controlado en un DataFrame, para demostrar la
    capacidad de detección del monitoreo (Escenario 2).

    Modificaciones aplicadas:
        - salario_cliente: se multiplica por 1.5 (simula inflación salarial
          o cambio en el segmento de clientes).
        - edad_cliente: se suma 8 años (simula envejecimiento de la cartera).
        - tipo_laboral: se invierte parcialmente la proporción (simula
          cambio en el perfil laboral de los nuevos solicitantes).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame al que inducir drift (típicamente el test set).

    Returns
    -------
    pd.DataFrame
        Copia del DataFrame con drift inducido.
    """
    df = df.copy()
    rng = np.random.default_rng(SEMILLA)

    # 1. Drift en salario: desplazamiento multiplicativo
    if "salario_cliente" in df.columns:
        df["salario_cliente"] = df["salario_cliente"] * 1.5

    # 2. Drift en edad: desplazamiento aditivo
    if "edad_cliente" in df.columns:
        df["edad_cliente"] = df["edad_cliente"] + 8

    # 3. Drift en tipo_laboral: cambiar proporciones.
    # Reasignamos ~30% de 'Empleado' a 'Independiente' aleatoriamente.
    if "tipo_laboral" in df.columns:
        mask_empleado = df["tipo_laboral"] == "Empleado"
        indices_empleado = df[mask_empleado].index
        n_cambiar = int(len(indices_empleado) * 0.30)
        if n_cambiar > 0:
            idx_cambiar = rng.choice(indices_empleado, size=n_cambiar, replace=False)
            df.loc[idx_cambiar, "tipo_laboral"] = "Independiente"

    return df


def analizar_drift_dataset(
    referencia: pd.DataFrame,
    nuevos: pd.DataFrame,
    nombre_escenario: str,
) -> pd.DataFrame:
    """
    Analiza el drift entre dos datasets, variable por variable.

    Aplica:
        - PSI a todas las variables (numéricas y categóricas codificadas).
        - KS a las variables numéricas.
        - Chi-cuadrado a las variables categóricas.

    Parameters
    ----------
    referencia : pd.DataFrame
        Dataset de referencia (train).
    nuevos : pd.DataFrame
        Dataset nuevo a comparar.
    nombre_escenario : str
        Nombre descriptivo del escenario (para logging).

    Returns
    -------
    pd.DataFrame
        Tabla con una fila por variable y columnas:
        variable, tipo, psi, psi_interpretacion, test_statistic,
        p_value, hay_drift_test.
    """
    logger.info("Analizando drift - escenario: %s", nombre_escenario)
    registros = []

    # Variables numéricas: PSI + KS
    for col in COLS_NUMERICAS:
        if col not in referencia.columns or col not in nuevos.columns:
            continue
        psi_res = calcular_psi(referencia[col], nuevos[col])
        ks_res = calcular_ks(referencia[col], nuevos[col])
        registros.append({
            "variable": col,
            "tipo": "numerica",
            "psi": round(psi_res["psi"], 4),
            "psi_interpretacion": psi_res["interpretacion"],
            "test": "KS",
            "test_statistic": round(ks_res["ks_statistic"], 4),
            "p_value": round(ks_res["p_value"], 4),
            "hay_drift_test": ks_res["hay_drift"],
        })

    # Variables categóricas: PSI (sobre frecuencias) + Chi-cuadrado
    for col in COLS_CATEGORICAS:
        if col not in referencia.columns or col not in nuevos.columns:
            continue
        # Para PSI categórico, convertimos a códigos numéricos por categoría
        chi2_res = calcular_chi2(referencia[col].astype(str), nuevos[col].astype(str))
        # PSI categórico: proporción por categoría
        psi_cat = calcular_psi_categorico(referencia[col].astype(str), nuevos[col].astype(str))
        registros.append({
            "variable": col,
            "tipo": "categorica",
            "psi": round(psi_cat["psi"], 4),
            "psi_interpretacion": psi_cat["interpretacion"],
            "test": "Chi2",
            "test_statistic": round(chi2_res["chi2_statistic"], 4),
            "p_value": round(chi2_res["p_value"], 4),
            "hay_drift_test": chi2_res["hay_drift"],
        })

    tabla = pd.DataFrame(registros)
    tabla = tabla.sort_values("psi", ascending=False).reset_index(drop=True)

    # Resumen en log
    n_drift_psi = (tabla["psi"] >= PSI_UMBRAL_MODERADO).sum()
    n_drift_test = tabla["hay_drift_test"].sum()
    logger.info("    Variables con PSI >= %.2f: %d de %d",
                PSI_UMBRAL_MODERADO, n_drift_psi, len(tabla))
    logger.info("    Variables con drift por test estadístico: %d de %d",
                n_drift_test, len(tabla))

    return tabla


def calcular_psi_categorico(referencia: pd.Series, nuevos: pd.Series) -> dict:
    """
    Calcula el PSI para una variable categórica usando las proporciones
    de cada categoría como bins.

    Parameters
    ----------
    referencia, nuevos : pd.Series
        Valores categóricos.

    Returns
    -------
    dict
        - 'psi': valor del índice.
        - 'interpretacion': categoría de drift.
    """
    categorias = sorted(set(referencia.unique()) | set(nuevos.unique()),
                        key=lambda x: str(x))
    ref_total = len(referencia)
    nue_total = len(nuevos)

    epsilon = 1e-4
    psi = 0.0
    for cat in categorias:
        ref_pct = (referencia == cat).sum() / ref_total
        nue_pct = (nuevos == cat).sum() / nue_total
        ref_pct = ref_pct if ref_pct > 0 else epsilon
        nue_pct = nue_pct if nue_pct > 0 else epsilon
        psi += (nue_pct - ref_pct) * np.log(nue_pct / ref_pct)

    return {"psi": float(psi), "interpretacion": interpretar_psi(float(psi))}


# =========================================================================
# 6. REPORTES Y PERSISTENCIA
# =========================================================================

def guardar_reportes_drift(
    tabla_sin_drift: pd.DataFrame,
    tabla_con_drift: pd.DataFrame,
) -> None:
    """
    Guarda los reportes de drift en CSV.

    Genera:
        - psi_report.csv: PSI por variable en ambos escenarios.
        - ks_report.csv: resultados de los tests estadísticos en ambos escenarios.

    Parameters
    ----------
    tabla_sin_drift, tabla_con_drift : pd.DataFrame
        Tablas de análisis de cada escenario.
    """
    # Etiquetar cada tabla con su escenario
    t1 = tabla_sin_drift.copy()
    t1.insert(0, "escenario", "sin_drift")
    t2 = tabla_con_drift.copy()
    t2.insert(0, "escenario", "con_drift_inducido")

    combinada = pd.concat([t1, t2], ignore_index=True)

    # Reporte PSI (columnas centradas en PSI)
    psi_cols = ["escenario", "variable", "tipo", "psi", "psi_interpretacion"]
    ruta_psi = RUTA_REPORTS / "psi_report.csv"
    combinada[psi_cols].to_csv(ruta_psi, index=False, encoding="utf-8")
    logger.info("Reporte PSI (CSV): %s", ruta_psi.relative_to(RUTA_RAIZ))

    # Reporte de tests estadísticos (KS / Chi2)
    test_cols = ["escenario", "variable", "tipo", "test", "test_statistic",
                 "p_value", "hay_drift_test"]
    ruta_ks = RUTA_REPORTS / "ks_report.csv"
    combinada[test_cols].to_csv(ruta_ks, index=False, encoding="utf-8")
    logger.info("Reporte de tests (CSV): %s", ruta_ks.relative_to(RUTA_RAIZ))


def graficar_drift_summary(
    tabla_sin_drift: pd.DataFrame,
    tabla_con_drift: pd.DataFrame,
) -> None:
    """
    Grafica el PSI por variable en ambos escenarios, con líneas de umbral.

    Genera un panel con dos subgráficos (sin drift / con drift) para
    comparación visual directa.

    Parameters
    ----------
    tabla_sin_drift, tabla_con_drift : pd.DataFrame
        Tablas de análisis de cada escenario.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

    for ax, tabla, titulo in zip(
        axes,
        [tabla_sin_drift, tabla_con_drift],
        ["Escenario 1: Sin drift (train vs test)",
         "Escenario 2: Con drift inducido"],
    ):
        datos = tabla.sort_values("psi", ascending=True)
        # Color según interpretación
        colores = datos["psi_interpretacion"].map({
            "sin_drift": "#5CB85C",
            "drift_moderado": "#F0AD4E",
            "drift_severo": "#D9534F",
        }).fillna("#5CB85C")

        ax.barh(datos["variable"], datos["psi"], color=colores, edgecolor="white")
        ax.axvline(x=PSI_UMBRAL_MODERADO, color="orange", linestyle="--",
                   linewidth=1.2, label=f"Umbral moderado ({PSI_UMBRAL_MODERADO})")
        ax.axvline(x=PSI_UMBRAL_SEVERO, color="red", linestyle="--",
                   linewidth=1.2, label=f"Umbral severo ({PSI_UMBRAL_SEVERO})")
        ax.set_xlabel("PSI")
        ax.set_title(titulo, fontsize=12)
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3, axis="x")

    plt.suptitle("Population Stability Index (PSI) por variable", fontsize=14)
    plt.tight_layout()
    ruta = RUTA_REPORTS / "drift_summary.png"
    plt.savefig(ruta, dpi=100, bbox_inches="tight")
    plt.close(fig)
    logger.info("Reporte visual: %s", ruta.relative_to(RUTA_RAIZ))


def graficar_distribuciones_drift(
    referencia: pd.DataFrame,
    nuevos_con_drift: pd.DataFrame,
    tabla_con_drift: pd.DataFrame,
    top_n: int = 3,
) -> None:
    """
    Compara las distribuciones (referencia vs nuevos) de las variables con
    mayor drift en el escenario inducido.

    Parameters
    ----------
    referencia : pd.DataFrame
        Dataset de referencia (train).
    nuevos_con_drift : pd.DataFrame
        Dataset con drift inducido.
    tabla_con_drift : pd.DataFrame
        Tabla de análisis del escenario con drift (ordenada por PSI desc).
    top_n : int
        Número de variables (con mayor PSI) a graficar.
    """
    # Tomar las top_n variables numéricas con mayor PSI
    top_vars = tabla_con_drift[
        tabla_con_drift["tipo"] == "numerica"
    ].head(top_n)["variable"].tolist()

    if not top_vars:
        logger.info("No hay variables numéricas con drift para graficar.")
        return

    fig, axes = plt.subplots(1, len(top_vars), figsize=(6 * len(top_vars), 5))
    if len(top_vars) == 1:
        axes = [axes]

    for ax, var in zip(axes, top_vars):
        ref_vals = referencia[var].dropna()
        nue_vals = nuevos_con_drift[var].dropna()
        ax.hist(ref_vals, bins=30, alpha=0.6, label="Referencia (train)",
                color="#5BC0DE", density=True, edgecolor="white")
        ax.hist(nue_vals, bins=30, alpha=0.6, label="Nuevos (con drift)",
                color="#D9534F", density=True, edgecolor="white")
        psi_var = tabla_con_drift[tabla_con_drift["variable"] == var]["psi"].iloc[0]
        ax.set_title(f"{var}\n(PSI={psi_var:.3f})", fontsize=11)
        ax.set_ylabel("Densidad")
        ax.legend()

    plt.suptitle("Comparación de distribuciones: referencia vs datos con drift",
                 fontsize=13)
    plt.tight_layout()
    ruta = RUTA_REPORTS / "distribuciones_drift.png"
    plt.savefig(ruta, dpi=100, bbox_inches="tight")
    plt.close(fig)
    logger.info("Reporte visual: %s", ruta.relative_to(RUTA_RAIZ))


def guardar_baseline(referencia: pd.DataFrame) -> None:
    """
    Guarda las estadísticas de referencia (baseline) en JSON.

    Este baseline representa la "foto" de la distribución de entrenamiento.
    En un sistema de producción real, se compararían los datos entrantes
    contra este baseline para detectar drift de forma continua.

    Parameters
    ----------
    referencia : pd.DataFrame
        Dataset de referencia (train).
    """
    baseline = {"variables_numericas": {}, "variables_categoricas": {}}

    for col in COLS_NUMERICAS:
        if col not in referencia.columns:
            continue
        serie = referencia[col].dropna()
        baseline["variables_numericas"][col] = {
            "media": round(float(serie.mean()), 4),
            "std": round(float(serie.std()), 4),
            "min": round(float(serie.min()), 4),
            "p25": round(float(serie.quantile(0.25)), 4),
            "p50": round(float(serie.quantile(0.50)), 4),
            "p75": round(float(serie.quantile(0.75)), 4),
            "max": round(float(serie.max()), 4),
        }

    for col in COLS_CATEGORICAS:
        if col not in referencia.columns:
            continue
        proporciones = referencia[col].astype(str).value_counts(normalize=True)
        baseline["variables_categoricas"][col] = {
            str(cat): round(float(pct), 4) for cat, pct in proporciones.items()
        }

    ruta = RUTA_MONITORING_ARTIFACTS / "drift_baseline.json"
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(baseline, f, ensure_ascii=False, indent=2)
    logger.info("Baseline guardado: %s", ruta.relative_to(RUTA_RAIZ))


# =========================================================================
# 7. FUNCIÓN PRINCIPAL
# =========================================================================
def main() -> None:
    """
    Orquesta el análisis de data drift en dos escenarios.

    Pasos (a desarrollar en las siguientes etapas):
        1. Cargar el dataset y reconstruir el split train/test.
        2. Escenario 1: drift entre train y test (sin drift esperado).
        3. Escenario 2: drift entre train y test con drift inducido.
        4. Generar reportes (CSV, PNG) y baseline JSON.
    """
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("Inicio del pipeline de monitoreo de data drift (V1.3.0)")
    logger.info("=" * 60)

    # Verificación: ¿existe el dataset?
    if not RUTA_DATASET.exists():
        logger.error("Dataset no encontrado: %s", RUTA_DATASET)
        raise FileNotFoundError(f"No se encontró el dataset: {RUTA_DATASET}")

    # Verificación: ¿existen las carpetas de destino?
    for carpeta in [RUTA_MONITORING_ARTIFACTS, RUTA_REPORTS]:
        carpeta.mkdir(parents=True, exist_ok=True)
        logger.info("Carpeta verificada: %s", carpeta.relative_to(RUTA_RAIZ))

    # ---------------------------------------------------------------------
    # Paso 1: Cargar dataset y aplicar limpieza mínima
    # ---------------------------------------------------------------------
    logger.info("Cargando dataset y aplicando limpieza mínima...")
    df = pd.read_excel(RUTA_DATASET)
    df = limpiar_minimo(df)
    logger.info("Dataset cargado y limpiado: %d filas x %d columnas",
                df.shape[0], df.shape[1])

    # ---------------------------------------------------------------------
    # Paso 2: Reconstruir el split train/test (mismo que ft_engineering)
    # ---------------------------------------------------------------------
    # Separamos X e y, descartando columnas excluidas, y replicamos el
    # split estratificado con la misma semilla.
    y = df[COL_TARGET]
    X = df.drop(columns=[c for c in COLS_EXCLUIDAS if c in df.columns])

    X_train, X_test = train_test_split(
        X, test_size=TEST_SIZE, stratify=y, random_state=SEMILLA
    )
    logger.info("Split reconstruido: referencia(train)=%d, nuevos(test)=%d",
                len(X_train), len(X_test))

    # ---------------------------------------------------------------------
    # Paso 3: Escenario 1 - train vs test (sin drift esperado)
    # ---------------------------------------------------------------------
    logger.info("-" * 60)
    tabla_sin_drift = analizar_drift_dataset(
        referencia=X_train,
        nuevos=X_test,
        nombre_escenario="Sin drift (train vs test)",
    )

    # ---------------------------------------------------------------------
    # Paso 4: Escenario 2 - train vs test con drift inducido
    # ---------------------------------------------------------------------
    logger.info("-" * 60)
    X_test_drift = inducir_drift(X_test)
    tabla_con_drift = analizar_drift_dataset(
        referencia=X_train,
        nuevos=X_test_drift,
        nombre_escenario="Con drift inducido (train vs test modificado)",
    )

    # ---------------------------------------------------------------------
    # Paso 5: Reportes y persistencia
    # ---------------------------------------------------------------------
    logger.info("-" * 60)
    logger.info("Generando reportes y baseline...")
    guardar_reportes_drift(tabla_sin_drift, tabla_con_drift)
    graficar_drift_summary(tabla_sin_drift, tabla_con_drift)
    graficar_distribuciones_drift(X_train, X_test_drift, tabla_con_drift)
    guardar_baseline(X_train)

    logger.info("-" * 60)
    logger.info("Pipeline de monitoreo V1.3.0 ejecutado completamente.")

    duracion = time.time() - t0
    logger.info("=" * 60)
    logger.info("Fin del pipeline. Tiempo total: %.1f segundos.", duracion)
    logger.info("=" * 60)


# =========================================================================
# 8. PUNTO DE ENTRADA
# =========================================================================
if __name__ == "__main__":
    main()
