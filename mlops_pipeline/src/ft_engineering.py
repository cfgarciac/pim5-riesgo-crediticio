"""
Módulo: ft_engineering.py
Propósito: Pipeline de ingeniería de características para el modelo de
           riesgo crediticio (target: Pago_atiempo).

Este script ejecutable toma el dataset crudo ``Base_de_datos.xlsx``, aplica
todas las transformaciones definidas en el EDA (limpieza, imputación,
capping, transformación logarítmica, encoding y escalado), divide en
train/test estratificado y persiste los artefactos resultantes.

Artefactos generados al ejecutar el script
------------------------------------------
- ``artifacts/data/X_train.parquet``
- ``artifacts/data/X_test.parquet``
- ``artifacts/data/y_train.parquet``
- ``artifacts/data/y_test.parquet``
- ``artifacts/transformers/pipeline_ft_engineering.pkl``
- ``artifacts/transformers/feature_names.json``
- ``reports/ft_engineering/distribuciones_antes_despues.png``
- ``reports/ft_engineering/target_train_test_split.png``
- ``reports/ft_engineering/correlacion_features_post.png``

Uso
---
Desde la raíz del repositorio:

    python mlops_pipeline/src/ft_engineering.py

Autor: Cristian García
Fecha de creación: 2026-05-26
Versión: V1.1.0 (esqueleto inicial)
"""

# =========================================================================
# 1. IMPORTS
# =========================================================================
from __future__ import annotations

# Biblioteca estándar
import json
import logging
from pathlib import Path

# Manipulación de datos
import numpy as np
import pandas as pd

# Visualización
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn: pipelines y transformadores
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
)

# Persistencia
import joblib


# =========================================================================
# 2. CONSTANTES Y CONFIGURACIÓN
# =========================================================================

# Rutas del proyecto (relativas a la raíz del repositorio)
# El script vive en mlops_pipeline/src/, las rutas suben dos niveles.
RUTA_RAIZ = Path(__file__).resolve().parents[2]
RUTA_DATASET = RUTA_RAIZ / "Base_de_datos.xlsx"
RUTA_ARTIFACTS = RUTA_RAIZ / "artifacts"
RUTA_REPORTS = RUTA_RAIZ / "reports" / "ft_engineering"

# Subcarpetas de artefactos
RUTA_DATA = RUTA_ARTIFACTS / "data"
RUTA_TRANSFORMERS = RUTA_ARTIFACTS / "transformers"

# Columnas especiales del dataset
COL_TARGET = "Pago_atiempo"
COL_FECHA = "fecha_prestamo"

# Lista explícita de categóricas codificadas (definidas en EDA - sección 1.4)
CATEGORICAS_CODIFICADAS = ["tipo_credito", "saldo_mora_codeudor"]

# Variables con anomalías por semántica de negocio (capping requerido)
VARIABLES_CON_ANOMALIAS = [
    "edad_cliente",
    "salario_cliente",
    "puntaje_datacredito",
    "puntaje",
    "total_otros_prestamos",
]

# Reglas de capping/NaN para cada variable con anomalías.
# Cada entrada define:
#   - lo, hi: límites inferior y superior (None si no aplica).
#   - accion_low/accion_high: 'cap' (recortar al límite) o 'nan' (convertir a NaN).
# Las decisiones se basan en conocimiento de dominio (no en percentiles
# del dataset), por lo que NO introducen data leakage entre train y test.
REGLAS_ANOMALIAS = {
    "edad_cliente": {
        "lo": 18, "hi": 100,
        "accion_low": "nan", "accion_high": "nan",
    },
    "salario_cliente": {
        "lo": 0, "hi": 100_000_000,
        "accion_low": "cap", "accion_high": "cap",
    },
    "puntaje_datacredito": {
        # Valores <=0 (incluye ceros que son código de "no reportado") -> NaN.
        # Valores >950 -> capear a 950.
        "lo": 1, "hi": 950,
        "accion_low": "nan", "accion_high": "cap",
    },
    "puntaje": {
        "lo": 0, "hi": 100,
        "accion_low": "cap", "accion_high": "cap",
    },
    "total_otros_prestamos": {
        "lo": 0, "hi": 500_000_000,
        "accion_low": "cap", "accion_high": "cap",
    },
}

# Categorías válidas para tendencia_ingresos (resto se agrupa como OTROS)
TENDENCIA_INGRESOS_VALIDAS = ["Creciente", "Estable", "Decreciente"]

# -----------------------------------------------------------------------------
# Clasificación de columnas para el ColumnTransformer (basada en el EDA).
# Estas listas dirigen qué transformaciones se aplican a cada grupo.
# -----------------------------------------------------------------------------

# Numéricas continuas con asimetría positiva alta (skew > 2): requieren log1p.
COLS_CONTINUAS_SESGADAS = [
    "capital_prestado",
    "salario_cliente",
    "total_otros_prestamos",
    "cuota_pactada",
    "saldo_mora",
    "saldo_total",
    "saldo_principal",
    "promedio_ingresos_datacredito",
    "creditos_sectorFinanciero",
    "creditos_sectorReal",
]

# Numéricas continuas sin asimetría alta: solo imputar + escalar.
COLS_CONTINUAS_NORMALES = [
    "edad_cliente",
    "puntaje",
    "puntaje_datacredito",
    "cant_creditosvigentes",
    "huella_consulta",
]

# Numéricas discretas (enteros con pocos valores únicos, pero son cantidades).
COLS_DISCRETAS = [
    "plazo_meses",
    "creditos_sectorCooperativo",
]

# Categóricas (texto + codificadas): imputar + one-hot encoding.
COLS_CATEGORICAS = [
    "tipo_credito",          # codificada (6 valores)
    "saldo_mora_codeudor",   # codificada (4 valores + nulos)
    "tipo_laboral",          # texto binaria
    "tendencia_ingresos",    # texto (ya limpia con DESCONOCIDO y OTROS)
]

# Columnas a descartar del pipeline (no son features predictoras).
COLS_DESCARTAR = ["fecha_prestamo"]

# Umbrales de configuración
UMBRAL_SKEW_LOG = 2.0   # asimetría a partir de la cual aplicar log1p
PERCENTIL_CAPPING = 0.99  # percentil para capping de anomalías
UMBRAL_NULOS_INDICADOR = 5.0  # % de nulos a partir del cual crear indicador
TEST_SIZE = 0.20
SEMILLA = 42


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
# 4. FUNCIONES DE LIMPIEZA
# =========================================================================

def cargar_dataset(ruta: Path = RUTA_DATASET) -> pd.DataFrame:
    """
    Carga el dataset crudo desde el archivo Excel.

    Parameters
    ----------
    ruta : Path
        Ruta absoluta al archivo ``Base_de_datos.xlsx``. Por defecto usa
        ``RUTA_DATASET`` definida en las constantes del módulo.

    Returns
    -------
    pd.DataFrame
        DataFrame con el dataset crudo, sin transformaciones.

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe en la ruta indicada.
    """
    if not ruta.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {ruta}")

    logger.info("Cargando dataset desde: %s", ruta.relative_to(RUTA_RAIZ))
    df = pd.read_excel(ruta)
    logger.info("Dataset cargado: %d filas x %d columnas", df.shape[0], df.shape[1])
    return df


def limpiar_tendencia_ingresos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia la columna 'tendencia_ingresos' según las reglas definidas en el EDA.

    El EDA detectó que esta columna mezcla:
        - Categorías válidas: 'Creciente', 'Estable', 'Decreciente'.
        - Valores numéricos sueltos como ruido (~1%).
        - Nulos (~27%).

    Reglas aplicadas:
        - NaN -> 'DESCONOCIDO'
        - Valor en TENDENCIA_INGRESOS_VALIDAS -> conservar
        - Cualquier otro valor -> 'OTROS'

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de entrada (no se modifica in-place).

    Returns
    -------
    pd.DataFrame
        Copia del DataFrame con la columna 'tendencia_ingresos' limpia.
    """
    df = df.copy()
    col = "tendencia_ingresos"

    def _mapear(valor):
        if pd.isna(valor):
            return "DESCONOCIDO"
        if valor in TENDENCIA_INGRESOS_VALIDAS:
            return valor
        return "OTROS"

    df[col] = df[col].apply(_mapear)

    # Logging informativo del resultado
    conteos = df[col].value_counts()
    logger.info("Limpieza de '%s' completada. Distribución resultante:", col)
    for cat, n in conteos.items():
        logger.info("    %-15s : %5d (%.2f%%)", cat, n, n / len(df) * 100)

    return df


def aplicar_capping_anomalias(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica capping y/o conversión a NaN para valores anómalos según las
    reglas definidas en REGLAS_ANOMALIAS.

    Las reglas se basan en conocimiento de dominio (rangos válidos
    estándar para cada variable), no en percentiles del dataset. Por
    tanto, esta función NO introduce data leakage al aplicarse antes
    del split train/test.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con las variables originales.

    Returns
    -------
    pd.DataFrame
        Copia del DataFrame con anomalías corregidas.
    """
    df = df.copy()
    logger.info("Aplicando reglas de capping/NaN sobre anomalías:")

    for col, reglas in REGLAS_ANOMALIAS.items():
        if col not in df.columns:
            logger.warning("    %s no está en el DataFrame; se omite.", col)
            continue

        serie = df[col]
        lo, hi = reglas["lo"], reglas["hi"]
        accion_low = reglas["accion_low"]
        accion_high = reglas["accion_high"]

        # Contar antes de modificar
        n_low = (serie < lo).sum()
        n_high = (serie > hi).sum()

        # Aplicar acción para valores bajos
        if accion_low == "nan":
            df.loc[df[col] < lo, col] = np.nan
        elif accion_low == "cap":
            df.loc[df[col] < lo, col] = lo

        # Aplicar acción para valores altos
        if accion_high == "nan":
            df.loc[df[col] > hi, col] = np.nan
        elif accion_high == "cap":
            df.loc[df[col] > hi, col] = hi

        logger.info(
            "    %-25s | rango[%s, %s] | low=%d (%s), high=%d (%s)",
            col, lo, hi, n_low, accion_low, n_high, accion_high
        )

    return df


def convertir_categoricas_codificadas_a_string(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte las columnas categóricas codificadas (CATEGORICAS_CODIFICADAS)
    a tipo string. Es necesario para que el OneHotEncoder pueda procesarlas
    junto con las categóricas de texto, sin mezclar tipos en una misma columna.

    Los NaN se preservan como NaN (no se convierten a la cadena 'nan').

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de entrada.

    Returns
    -------
    pd.DataFrame
        Copia del DataFrame con las columnas codificadas como string.
    """
    df = df.copy()
    for col in CATEGORICAS_CODIFICADAS:
        if col in df.columns:
            # Convertir a string preservando NaN
            df[col] = df[col].apply(
                lambda v: str(int(v)) if pd.notna(v) and float(v).is_integer()
                else (str(v) if pd.notna(v) else np.nan)
            )
    logger.info(
        "Categóricas codificadas convertidas a string: %s",
        CATEGORICAS_CODIFICADAS
    )
    return df


# =========================================================================
# 5. TRANSFORMERS PERSONALIZADOS
# =========================================================================

class NullIndicatorTransformer(BaseEstimator, TransformerMixin):
    """
    Crea columnas indicadoras binarias para las columnas con un porcentaje
    de nulos mayor o igual a un umbral.

    Para cada columna que supera el umbral, se agrega una columna nueva
    ``<columna>_fue_imputado`` con valor 1 donde el valor original era nulo
    y 0 en caso contrario. Las columnas originales se conservan intactas
    (otra etapa del pipeline se encargará de imputarlas).

    Parameters
    ----------
    umbral_pct : float
        Porcentaje (0-100) de nulos a partir del cual se crea un indicador.
        Por defecto 5.0.

    Attributes
    ----------
    columnas_con_indicador_ : list[str]
        Lista de columnas para las que se creó un indicador (aprendido en fit).
    """

    def __init__(self, umbral_pct: float = 5.0):
        self.umbral_pct = umbral_pct

    def fit(self, X: pd.DataFrame, y=None) -> "NullIndicatorTransformer":
        # Guardamos los nombres de columnas de entrada (requisito sklearn)
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        # Aprendemos qué columnas requieren indicador SOLO con los datos
        # de entrenamiento (X aquí será X_train).
        pct_nulos = X.isnull().mean() * 100
        self.columnas_con_indicador_ = pct_nulos[
            pct_nulos >= self.umbral_pct
        ].index.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.columnas_con_indicador_:
            if col in X.columns:
                X[f"{col}_fue_imputado"] = X[col].isnull().astype(int)
        return X

    def get_feature_names_out(self, input_features=None):
        """Devuelve los nombres de las columnas resultantes (compatibilidad sklearn)."""
        if input_features is None:
            base = list(self.feature_names_in_) if hasattr(self, "feature_names_in_") else []
        else:
            base = list(input_features)
        extras = [f"{c}_fue_imputado" for c in self.columnas_con_indicador_]
        return np.array(base + extras, dtype=object)


class Log1pTransformer(BaseEstimator, TransformerMixin):
    """
    Aplica la transformación ``log1p(x) = log(1 + x)`` a las columnas
    indicadas. Preserva los valores NaN sin modificarlos.

    La transformación es útil para reducir la asimetría positiva de
    variables financieras. ``log1p`` se prefiere a ``log`` porque maneja
    correctamente el valor cero (``log1p(0) = 0``).

    Parameters
    ----------
    columnas : list[str] | None
        Lista de columnas a las que aplicar log1p. Si es None, se aplica
        a todas las columnas numéricas que cumplan condición en fit.
    """

    def __init__(self, columnas: list | None = None):
        self.columnas = columnas

    def fit(self, X: pd.DataFrame, y=None) -> "Log1pTransformer":
        # Guardamos los nombres de columnas de entrada (requisito sklearn)
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.array(X.columns, dtype=object)
        # Validar que las columnas existen en X (si se pasaron explícitas)
        if self.columnas is not None:
            faltantes = [c for c in self.columnas if c not in X.columns]
            if faltantes:
                raise ValueError(
                    f"Columnas no encontradas para Log1pTransformer: {faltantes}"
                )
            self.columnas_ = list(self.columnas)
        else:
            # Modo automático: aplicar a todas las numéricas
            self.columnas_ = X.select_dtypes(include=[np.number]).columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.columnas_:
            if col in X.columns:
                # log1p preserva NaN automáticamente; no propaga errores
                # con valores negativos (devolvería NaN), pero todas
                # nuestras columnas ya pasaron por capping y son >= 0.
                X[col] = np.log1p(X[col])
        return X

    def get_feature_names_out(self, input_features=None):
        """Devuelve los nombres de las columnas resultantes (compatibilidad sklearn).
        Log1p no agrega ni quita columnas: los nombres de salida son los mismos
        que los de entrada."""
        if input_features is None:
            return np.array(self.feature_names_in_, dtype=object) \
                if hasattr(self, "feature_names_in_") else np.array([])
        return np.array(input_features, dtype=object)


# =========================================================================
# 6. CONSTRUCCIÓN DEL PIPELINE SKLEARN
# =========================================================================

def construir_pipeline_ft_engineering() -> Pipeline:
    """
    Construye el pipeline sklearn maestro para feature engineering.

    Arquitectura del pipeline:

        Pipeline maestro:
        1) NullIndicatorTransformer (agrega columnas indicadoras de nulos).
        2) ColumnTransformer (transforma columnas por grupo):
            - continuas_sesgadas    : Impute(median) -> Log1p -> StandardScaler
            - continuas_normales    : Impute(median) -> StandardScaler
            - discretas             : Impute(median) -> StandardScaler
            - categoricas           : Impute(constant 'DESCONOCIDO') -> OneHot
            - indicadores_nulos     : Pasthrough (ya son 0/1)
            - fecha_prestamo        : drop

    El pipeline devuelve un DataFrame de pandas con nombres de columna
    legibles (gracias a set_output(transform='pandas') de sklearn 1.2+).

    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline sin ajustar (debe llamarse fit antes de transform).
    """
    # --- Sub-pipeline para continuas sesgadas ---
    # Orden: imputar nulos -> log1p (reduce asimetría) -> escalar.
    pipe_continuas_sesgadas = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("log1p", Log1pTransformer()),  # auto-detecta numéricas en fit
        ("scaler", StandardScaler()),
    ])

    # --- Sub-pipeline para continuas normales (sin log1p) ---
    pipe_continuas_normales = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # --- Sub-pipeline para numéricas discretas ---
    # Mismo tratamiento que continuas normales: imputar mediana + escalar.
    pipe_discretas = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # --- Sub-pipeline para categóricas ---
    # Impute con valor constante 'DESCONOCIDO' para los nulos remanentes,
    # luego one-hot con drop='first' (TD31) y handle_unknown='ignore' (TD32).
    pipe_categoricas = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="DESCONOCIDO")),
        ("onehot", OneHotEncoder(
            drop="first",
            handle_unknown="ignore",
            sparse_output=False,
        )),
    ])

    # --- ColumnTransformer: aplica cada sub-pipeline al grupo correcto ---
    column_transformer = ColumnTransformer(
        transformers=[
            ("continuas_sesgadas", pipe_continuas_sesgadas, COLS_CONTINUAS_SESGADAS),
            ("continuas_normales", pipe_continuas_normales, COLS_CONTINUAS_NORMALES),
            ("discretas", pipe_discretas, COLS_DISCRETAS),
            ("categoricas", pipe_categoricas, COLS_CATEGORICAS),
            # Columnas a descartar (fecha + columnas que no estén listadas)
        ],
        remainder="passthrough",  # las columnas no listadas pasan tal cual
        verbose_feature_names_out=False,  # nombres limpios sin prefijos
    )

    # --- Pipeline maestro ---
    # Paso 1: agregar indicadores de nulos (antes del ColumnTransformer
    # para que los indicadores ya existan cuando 'passthrough' los recoja).
    # Paso 2: aplicar ColumnTransformer.
    pipeline_maestro = Pipeline(steps=[
        ("null_indicators", NullIndicatorTransformer(umbral_pct=UMBRAL_NULOS_INDICADOR)),
        ("column_transformer", column_transformer),
    ])

    # Configurar salida como pandas DataFrame con nombres de columna (TD33)
    pipeline_maestro.set_output(transform="pandas")

    return pipeline_maestro


# =========================================================================
# 7. PERSISTENCIA Y REPORTES VISUALES
# =========================================================================

def guardar_datasets_transformados(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """
    Guarda los datasets transformados en formato Parquet.

    Los archivos se persisten en ``artifacts/data/``:
        - X_train.parquet
        - X_test.parquet
        - y_train.parquet
        - y_test.parquet

    Parameters
    ----------
    X_train, X_test : pd.DataFrame
        Features transformadas para entrenamiento y prueba.
    y_train, y_test : pd.Series
        Target para entrenamiento y prueba.
    """
    # Convertir y a DataFrame para uniformidad en lectura posterior
    y_train_df = y_train.to_frame(name=COL_TARGET)
    y_test_df = y_test.to_frame(name=COL_TARGET)

    X_train.to_parquet(RUTA_DATA / "X_train.parquet", index=True, compression="snappy")
    X_test.to_parquet(RUTA_DATA / "X_test.parquet", index=True, compression="snappy")
    y_train_df.to_parquet(RUTA_DATA / "y_train.parquet", index=True, compression="snappy")
    y_test_df.to_parquet(RUTA_DATA / "y_test.parquet", index=True, compression="snappy")

    logger.info("Datasets transformados guardados en %s:", RUTA_DATA.relative_to(RUTA_RAIZ))
    logger.info("    X_train.parquet (%d filas)", len(X_train))
    logger.info("    X_test.parquet  (%d filas)", len(X_test))
    logger.info("    y_train.parquet (%d filas)", len(y_train))
    logger.info("    y_test.parquet  (%d filas)", len(y_test))


def guardar_pipeline(pipeline: Pipeline, feature_names: list[str]) -> None:
    """
    Guarda el pipeline entrenado y los nombres de features.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Pipeline entrenado (ya pasó por fit).
    feature_names : list[str]
        Lista ordenada de nombres de columnas de salida.
    """
    # Pipeline serializado con joblib (más eficiente que pickle para sklearn)
    ruta_pipeline = RUTA_TRANSFORMERS / "pipeline_ft_engineering.pkl"
    joblib.dump(pipeline, ruta_pipeline)
    logger.info("Pipeline serializado: %s", ruta_pipeline.relative_to(RUTA_RAIZ))

    # Lista de features en JSON (útil para la API y el dashboard)
    ruta_features = RUTA_TRANSFORMERS / "feature_names.json"
    with open(ruta_features, "w", encoding="utf-8") as f:
        json.dump(feature_names, f, ensure_ascii=False, indent=2)
    logger.info("Nombres de features guardados: %s", ruta_features.relative_to(RUTA_RAIZ))


def graficar_distribuciones_antes_despues(
    df_raw: pd.DataFrame,
    X_train_transformado: pd.DataFrame,
) -> None:
    """
    Compara la distribución de las variables sesgadas antes y después
    de aplicar el pipeline (log1p + escalado).

    Parameters
    ----------
    df_raw : pd.DataFrame
        DataFrame original tras limpieza, antes del pipeline.
    X_train_transformado : pd.DataFrame
        Features transformadas (solo se usan las que corresponden a las
        columnas sesgadas).
    """
    # Tomar las columnas sesgadas que existen en ambos lados
    cols_a_comparar = [
        c for c in COLS_CONTINUAS_SESGADAS
        if c in df_raw.columns and c in X_train_transformado.columns
    ]
    n_cols = len(cols_a_comparar)
    n_filas = int(np.ceil(n_cols / 2))

    fig, axes = plt.subplots(n_filas, 4, figsize=(16, n_filas * 3))

    for i, col in enumerate(cols_a_comparar):
        fila = i // 2
        col_idx = (i % 2) * 2  # cada variable ocupa 2 columnas (antes / después)

        # Antes (DataFrame crudo)
        serie_raw = df_raw[col].dropna()
        axes[fila, col_idx].hist(serie_raw, bins=30, color="#D9534F", edgecolor="white")
        axes[fila, col_idx].set_title(f"{col} - ANTES", fontsize=9)
        axes[fila, col_idx].set_ylabel("Frecuencia")

        # Después (post-pipeline)
        serie_t = X_train_transformado[col].dropna()
        axes[fila, col_idx + 1].hist(serie_t, bins=30, color="#5BC0DE", edgecolor="white")
        axes[fila, col_idx + 1].set_title(f"{col} - DESPUÉS (log1p + escalado)", fontsize=9)

    plt.suptitle(
        "Distribución de variables sesgadas: antes vs después del pipeline",
        fontsize=13, y=1.00
    )
    plt.tight_layout()

    ruta = RUTA_REPORTS / "distribuciones_antes_despues.png"
    plt.savefig(ruta, dpi=100, bbox_inches="tight")
    plt.close(fig)
    logger.info("Reporte visual guardado: %s", ruta.relative_to(RUTA_RAIZ))


def graficar_target_split(y_train: pd.Series, y_test: pd.Series) -> None:
    """
    Visualiza la distribución del target en train y test para verificar
    que el split estratificado preservó las proporciones.

    Parameters
    ----------
    y_train, y_test : pd.Series
        Targets de entrenamiento y prueba.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, y, nombre in zip(axes, [y_train, y_test], ["Train", "Test"]):
        conteos = y.value_counts().sort_index()
        proporciones = (conteos / len(y) * 100).round(2)
        colores = ["#D9534F", "#5BC0DE"]
        ax.bar(conteos.index.astype(str), conteos.values, color=colores, edgecolor="white")
        ax.set_title(f"{nombre} (n={len(y):,})")
        ax.set_xlabel(f"{COL_TARGET}")
        ax.set_ylabel("Frecuencia")
        for i, (v, p) in enumerate(zip(conteos.values, proporciones.values)):
            ax.text(i, v + 30, f"{v:,}\n({p}%)", ha="center", fontsize=10)

    plt.suptitle("Distribución del target en train vs test (split estratificado)", fontsize=13)
    plt.tight_layout()

    ruta = RUTA_REPORTS / "target_train_test_split.png"
    plt.savefig(ruta, dpi=100, bbox_inches="tight")
    plt.close(fig)
    logger.info("Reporte visual guardado: %s", ruta.relative_to(RUTA_RAIZ))


def graficar_correlacion_features_post(X_train_transformado: pd.DataFrame) -> None:
    """
    Heatmap de correlación de las features finales tras el pipeline.

    Útil para detectar redundancias generadas por el encoding
    (por ejemplo, columnas one-hot del mismo grupo categórico que pueden
    quedar altamente correlacionadas).

    Parameters
    ----------
    X_train_transformado : pd.DataFrame
        Features de entrenamiento ya transformadas.
    """
    matriz_corr = X_train_transformado.corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        matriz_corr,
        annot=False,  # 33x33 = demasiados números para anotar
        cmap="RdBu_r",
        center=0,
        square=True,
        linewidths=0.2,
        cbar_kws={"shrink": 0.7},
        ax=ax,
    )
    ax.set_title("Correlación entre features finales (post pipeline)", fontsize=13)
    plt.tight_layout()

    ruta = RUTA_REPORTS / "correlacion_features_post.png"
    plt.savefig(ruta, dpi=100, bbox_inches="tight")
    plt.close(fig)
    logger.info("Reporte visual guardado: %s", ruta.relative_to(RUTA_RAIZ))


# =========================================================================
# 8. FUNCIÓN PRINCIPAL
# =========================================================================
def main() -> None:
    """
    Orquesta el pipeline completo de ingeniería de características.

    Pasos a ejecutar (a desarrollar en las siguientes etapas):
        1. Cargar dataset crudo.
        2. Aplicar limpieza inicial (tendencia_ingresos, capping).
        3. Separar features (X) y target (y).
        4. Split estratificado train/test.
        5. Construir y ajustar pipeline sklearn sobre X_train.
        6. Transformar X_train y X_test.
        7. Guardar artefactos (datasets, pipeline, feature names).
        8. Generar reportes visuales.
    """
    logger.info("=" * 60)
    logger.info("Inicio del pipeline de feature engineering (V1.1.0)")
    logger.info("=" * 60)

    # Verificación básica: ¿existe el dataset?
    if not RUTA_DATASET.exists():
        logger.error("Dataset no encontrado en %s", RUTA_DATASET)
        raise FileNotFoundError(f"No se encontró el dataset: {RUTA_DATASET}")

    # Verificación básica: ¿existen las carpetas de destino?
    for carpeta in [RUTA_DATA, RUTA_TRANSFORMERS, RUTA_REPORTS]:
        carpeta.mkdir(parents=True, exist_ok=True)
        logger.info("Carpeta verificada: %s", carpeta.relative_to(RUTA_RAIZ))

    # Paso 1: Cargar dataset crudo
    df = cargar_dataset()

    # Paso 2: Limpieza inicial
    df = limpiar_tendencia_ingresos(df)
    df = aplicar_capping_anomalias(df)
    df = convertir_categoricas_codificadas_a_string(df)

    # Reportar estado tras la limpieza
    n_nulos_post = df.isnull().sum().sum()
    logger.info(
        "Tras limpieza inicial: %d filas, %d columnas, %d valores nulos en total.",
        df.shape[0], df.shape[1], n_nulos_post
    )

    # ---------------------------------------------------------------------
    # Paso 3: Separar features (X) y target (y), y descartar columnas
    # que no deben entrar al pipeline.
    # ---------------------------------------------------------------------
    y = df[COL_TARGET].copy()
    X = df.drop(columns=[COL_TARGET] + COLS_DESCARTAR)
    logger.info(
        "Separación X/y: X tiene %d columnas, y tiene %d valores.",
        X.shape[1], len(y)
    )

    # ---------------------------------------------------------------------
    # Paso 4: Split estratificado train/test (TD18).
    # Estratificado por y para preservar la proporción 95/5 del target.
    # ---------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=SEMILLA,
    )
    logger.info(
        "Split realizado: X_train=%d, X_test=%d (test_size=%.0f%%, stratify=y)",
        len(X_train), len(X_test), TEST_SIZE * 100
    )
    logger.info(
        "    Proporción target en train: %.4f | en test: %.4f",
        y_train.mean(), y_test.mean()
    )

    # ---------------------------------------------------------------------
    # Paso 5: Construir, ajustar (fit) y aplicar (transform) el pipeline.
    # ---------------------------------------------------------------------
    pipeline = construir_pipeline_ft_engineering()
    logger.info("Pipeline construido. Iniciando fit sobre X_train...")

    # CRÍTICO: fit SOLO con X_train. Esto evita data leakage.
    pipeline.fit(X_train, y_train)
    logger.info("Pipeline ajustado.")

    # Aplicar el mismo pipeline a train y test.
    X_train_t = pipeline.transform(X_train)
    X_test_t = pipeline.transform(X_test)
    logger.info(
        "Transformación aplicada: X_train_t=%s, X_test_t=%s",
        X_train_t.shape, X_test_t.shape
    )
    logger.info(
        "    Features generadas (%d):", X_train_t.shape[1]
    )
    for i, col in enumerate(X_train_t.columns, start=1):
        logger.info("        %2d. %s", i, col)

    # Verificación: ¿quedó algún nulo tras el pipeline?
    n_nulos_train = X_train_t.isnull().sum().sum()
    n_nulos_test = X_test_t.isnull().sum().sum()
    if n_nulos_train == 0 and n_nulos_test == 0:
        logger.info("Verificación de nulos OK: ningún NaN en X_train_t ni X_test_t.")
    else:
        logger.warning(
            "ATENCIÓN: quedan %d nulos en X_train_t y %d en X_test_t.",
            n_nulos_train, n_nulos_test
        )

    # ---------------------------------------------------------------------
    # Paso 6: Persistencia de artefactos
    # ---------------------------------------------------------------------
    logger.info("-" * 60)
    logger.info("Iniciando persistencia de artefactos...")
    guardar_datasets_transformados(X_train_t, X_test_t, y_train, y_test)
    guardar_pipeline(pipeline, feature_names=X_train_t.columns.tolist())

    # ---------------------------------------------------------------------
    # Paso 7: Reportes visuales
    # ---------------------------------------------------------------------
    logger.info("-" * 60)
    logger.info("Generando reportes visuales...")

    # Para el reporte antes/después necesitamos un df "raw" alineado con
    # el split de train (antes de pasar por el pipeline).
    df_raw_train = X_train.copy()
    graficar_distribuciones_antes_despues(df_raw_train, X_train_t)
    graficar_target_split(y_train, y_test)
    graficar_correlacion_features_post(X_train_t)

    logger.info("-" * 60)
    logger.info("Pipeline V1.1.0 ejecutado completamente.")

    logger.info("=" * 60)
    logger.info("Fin del pipeline.")
    logger.info("=" * 60)


# =========================================================================
# 9. PUNTO DE ENTRADA
# =========================================================================
if __name__ == "__main__":
    main()
