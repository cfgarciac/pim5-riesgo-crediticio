"""
Módulo: model_training_evaluation.py
Propósito: Entrenamiento, comparación y selección del modelo predictivo
           óptimo para riesgo crediticio (target: Pago_atiempo).

Este script ejecutable consume los artefactos generados por
``ft_engineering.py`` (datasets transformados en Parquet y pipeline serializado),
entrena 4 modelos supervisados con RandomizedSearchCV, selecciona el mejor
según métricas de negocio (ROC-AUC, F1, Precision-Recall), optimiza sus
hiperparámetros con Optuna, ajusta el umbral de decisión por análisis de
costo, y persiste el modelo final junto con sus métricas y artefactos
visuales.

Modelos entrenados
------------------
1. Regresión Logística (baseline lineal, class_weight='balanced').
2. Random Forest (class_weight='balanced').
3. XGBoost (scale_pos_weight calculado del desbalance).
4. LightGBM (class_weight='balanced').

Artefactos generados al ejecutar el script
------------------------------------------
- ``artifacts/models/modelo_ganador.pkl``
- ``artifacts/models/threshold_optimo.json``
- ``artifacts/models/metricas_finales.json``
- ``reports/model_training/tabla_comparativa_modelos.csv``
- ``reports/model_training/tabla_comparativa_modelos.png``
- ``reports/model_training/roc_curve_comparativa.png``
- ``reports/model_training/precision_recall_curve_comparativa.png``
- ``reports/model_training/matriz_confusion_ganador_threshold_default.png``
- ``reports/model_training/matriz_confusion_ganador_threshold_optimizado.png``
- ``reports/model_training/feature_importance.png``
- ``reports/model_training/curva_costo_vs_threshold.png``
- ``reports/model_training/optuna_history.png``

Uso
---
Desde la raíz del repositorio (requiere haber ejecutado antes
``ft_engineering.py``):

    python mlops_pipeline/src/model_training_evaluation.py

Autor: Cristian García
Fecha de creación: 2026-05-27
Versión: V1.0.1 (esqueleto inicial)
"""

# =========================================================================
# 1. IMPORTS
# =========================================================================
from __future__ import annotations

# Biblioteca estándar
import json
import logging
import time
from pathlib import Path

# Manipulación de datos
import numpy as np
import pandas as pd

# Visualización
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn: modelos y utilidades
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    average_precision_score,
)

# Modelos avanzados
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Optimización bayesiana
import optuna

# Persistencia
import joblib


# =========================================================================
# 2. CONSTANTES Y CONFIGURACIÓN
# =========================================================================

# Rutas del proyecto (relativas a la raíz)
RUTA_RAIZ = Path(__file__).resolve().parents[2]

# Inputs (artefactos generados por ft_engineering.py)
RUTA_ARTIFACTS = RUTA_RAIZ / "artifacts"
RUTA_DATA = RUTA_ARTIFACTS / "data"
RUTA_TRANSFORMERS = RUTA_ARTIFACTS / "transformers"

# Outputs (este script genera estos)
RUTA_MODELS = RUTA_ARTIFACTS / "models"
RUTA_REPORTS = RUTA_RAIZ / "reports" / "model_training"

# Columna objetivo
COL_TARGET = "Pago_atiempo"

# Configuración de entrenamiento
SEMILLA = 42
N_SPLITS_CV = 5             # k de StratifiedKFold (DM1)
SCORING_METRIC = "roc_auc"  # métrica de scoring en RandomizedSearchCV (DM2)
N_ITER_RANDOMIZED = 30      # iteraciones de RandomizedSearchCV por modelo (DM3)
N_TRIALS_OPTUNA = 50        # trials de Optuna sobre el ganador (DM4)
N_JOBS = -1                 # paralelización: usar todos los cores disponibles

# Threshold tuning (DM5)
# Ratio de costos alineado con el desbalance del dataset (~20:1).
# Dejar pasar un impago (aprobar a quien no paga) es el evento grave.
COSTO_IMPAGO_NO_DETECTADO = 20  # real=0 (no paga) pero el modelo lo aprueba (pred=1)
COSTO_BUEN_CLIENTE_RECHAZADO = 1  # real=1 (paga) pero el modelo lo rechaza (pred=0)


# =========================================================================
# 3. CONFIGURACIÓN DE LOGGING
# =========================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Suprimir warnings de Optuna para limpieza del output
optuna.logging.set_verbosity(optuna.logging.WARNING)


# =========================================================================
# 4. CARGA DE ARTEFACTOS
# =========================================================================

def cargar_datasets_transformados() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Carga los 4 datasets transformados (Parquet) generados por
    ``ft_engineering.py``.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test) donde X son DataFrames con
        las features transformadas y y son Series con el target.
    """
    X_train = pd.read_parquet(RUTA_DATA / "X_train.parquet")
    X_test = pd.read_parquet(RUTA_DATA / "X_test.parquet")
    # y se guardó como DataFrame de una columna; tomamos la Serie.
    y_train = pd.read_parquet(RUTA_DATA / "y_train.parquet")[COL_TARGET]
    y_test = pd.read_parquet(RUTA_DATA / "y_test.parquet")[COL_TARGET]

    logger.info("Datasets transformados cargados:")
    logger.info("    X_train: %s | y_train: %d valores", X_train.shape, len(y_train))
    logger.info("    X_test:  %s | y_test:  %d valores", X_test.shape, len(y_test))
    logger.info("    Tasa de Pago_atiempo=1 en train: %.4f", y_train.mean())
    logger.info("    Tasa de Pago_atiempo=1 en test:  %.4f", y_test.mean())

    return X_train, X_test, y_train, y_test


# Nota: la carga del pipeline serializado (``pipeline_ft_engineering.pkl``)
# se posterga al script de despliegue (Avance #4). Aquí no se necesita
# porque ya tenemos los datasets transformados en Parquet. Cargarlo
# requeriría importar los Transformers personalizados (NullIndicatorTransformer,
# Log1pTransformer) que viven en ft_engineering.py, lo cual generaría un
# acoplamiento innecesario entre módulos.


# =========================================================================
# 5. DEFINICIÓN DE MODELOS BASE Y ESPACIOS DE BÚSQUEDA
# =========================================================================

def calcular_scale_pos_weight(y: pd.Series) -> float:
    """
    Calcula scale_pos_weight para XGBoost dado el desbalance del target.

    En nuestro problema, ``Pago_atiempo=1`` es la clase mayoritaria
    (~95%) y ``Pago_atiempo=0`` (impago) es la minoritaria (~5%). Sin
    embargo, desde la perspectiva del negocio, los clientes que NO pagan
    son los de mayor interés (mayor riesgo crediticio).

    El parámetro ``scale_pos_weight`` en XGBoost pondera la clase 1.
    Para compensar el desbalance, usamos la fórmula estándar:
        scale_pos_weight = n_negativos / n_positivos

    Esto equilibra la importancia de ambas clases durante el entrenamiento.

    Returns
    -------
    float
        scale_pos_weight calculado.
    """
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    # Fórmula estándar: peso de la clase 1 = n_clase_0 / n_clase_1.
    # Cuando la clase 1 es mayoritaria (como aquí), el resultado es < 1
    # y se le da MENOS peso a la mayoritaria, equivalente a dar más
    # peso relativo a la minoritaria.
    return n_neg / n_pos if n_pos > 0 else 1.0


def obtener_modelos_base(y_train: pd.Series) -> dict:
    """
    Define los 4 modelos base con configuración para manejo de desbalance.

    Cada modelo se configura con:
        - LR y RF: class_weight='balanced' (sklearn lo maneja automáticamente).
        - XGBoost: scale_pos_weight calculado del ratio negativos/positivos.
        - LightGBM: class_weight='balanced'.

    Parameters
    ----------
    y_train : pd.Series
        Target de entrenamiento (usado para calcular scale_pos_weight).

    Returns
    -------
    dict
        Diccionario {nombre_modelo: modelo_instanciado}.
    """
    spw = calcular_scale_pos_weight(y_train)
    logger.info("scale_pos_weight calculado para XGBoost: %.4f", spw)

    modelos = {
        "LogisticRegression": LogisticRegression(
            random_state=SEMILLA,
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear",  # compatible con l1 y l2
        ),
        "RandomForest": RandomForestClassifier(
            random_state=SEMILLA,
            class_weight="balanced",
            n_jobs=N_JOBS,
        ),
        "XGBoost": XGBClassifier(
            random_state=SEMILLA,
            scale_pos_weight=spw,
            eval_metric="logloss",
            n_jobs=N_JOBS,
            use_label_encoder=False,  # silenciar warning de versiones antiguas
            verbosity=0,
        ),
        "LightGBM": LGBMClassifier(
            random_state=SEMILLA,
            class_weight="balanced",
            n_jobs=N_JOBS,
            verbose=-1,  # silenciar logs internos
        ),
    }
    return modelos


def obtener_espacios_busqueda() -> dict:
    """
    Define los espacios de hiperparámetros para RandomizedSearchCV.

    Los rangos están escogidos para cubrir lo que típicamente funciona
    bien en problemas tabulares de tamaño medio (~10K filas, ~30 features).

    Returns
    -------
    dict
        Diccionario {nombre_modelo: dict_de_distribuciones}.
    """
    espacios = {
        "LogisticRegression": {
            "C": [0.001, 0.01, 0.1, 1, 10, 100],
            "penalty": ["l1", "l2"],
        },
        "RandomForest": {
            "n_estimators": [50, 100, 150, 200, 250, 300],
            "max_depth": [3, 5, 8, 10, 15, 20, None],
            "min_samples_split": [2, 5, 10, 15, 20],
            "min_samples_leaf": [1, 2, 4, 6, 10],
            "max_features": ["sqrt", "log2"],
        },
        "XGBoost": {
            "n_estimators": [100, 200, 300, 400, 500],
            "max_depth": [3, 4, 5, 6, 7, 8, 10],
            "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2, 0.3],
            "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            "min_child_weight": [1, 3, 5, 7, 10],
        },
        "LightGBM": {
            "n_estimators": [100, 200, 300, 400, 500],
            "max_depth": [3, 5, 7, 10, 15, -1],
            "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2, 0.3],
            "num_leaves": [20, 31, 50, 70, 100],
            "min_child_samples": [5, 10, 20, 30, 50],
        },
    }
    return espacios


# =========================================================================
# 6. ENTRENAMIENTO CON RANDOMIZED SEARCH
# =========================================================================

def entrenar_modelo_con_randomized_search(
    nombre: str,
    modelo,
    espacio_busqueda: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
    n_iter: int = N_ITER_RANDOMIZED,
    scoring: str = SCORING_METRIC,
):
    """
    Entrena un modelo aplicando RandomizedSearchCV sobre el espacio de
    hiperparámetros indicado.

    Parameters
    ----------
    nombre : str
        Nombre del modelo (para logging).
    modelo : sklearn estimator
        Modelo base ya configurado (con class_weight, scale_pos_weight, etc.).
    espacio_busqueda : dict
        Diccionario de hiperparámetros y sus distribuciones / listas.
    X_train : pd.DataFrame
        Features de entrenamiento.
    y_train : pd.Series
        Target de entrenamiento.
    cv : StratifiedKFold
        Estrategia de validación cruzada.
    n_iter : int
        Número de combinaciones aleatorias a explorar.
    scoring : str
        Métrica a optimizar (por defecto 'roc_auc').

    Returns
    -------
    dict
        Diccionario con:
            - 'best_estimator': el modelo entrenado con los mejores hiperparámetros.
            - 'best_params': los hiperparámetros ganadores.
            - 'best_cv_score': mejor score CV (validación cruzada).
            - 'cv_results': resultados completos del search (para análisis).
            - 'tiempo_segundos': duración del entrenamiento.
    """
    logger.info("-" * 60)
    logger.info("Entrenando %s con RandomizedSearchCV (n_iter=%d)...",
                nombre, n_iter)
    t0 = time.time()

    search = RandomizedSearchCV(
        estimator=modelo,
        param_distributions=espacio_busqueda,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        n_jobs=N_JOBS,
        random_state=SEMILLA,
        verbose=0,
        return_train_score=False,
    )

    search.fit(X_train, y_train)
    duracion = time.time() - t0

    logger.info(
        "    Tiempo de entrenamiento: %.1f segundos (%.1f min)",
        duracion, duracion / 60
    )
    logger.info("    Mejor %s (CV): %.4f", scoring, search.best_score_)
    logger.info("    Mejores hiperparámetros:")
    for k, v in sorted(search.best_params_.items()):
        logger.info("        %s = %s", k, v)

    return {
        "best_estimator": search.best_estimator_,
        "best_params": search.best_params_,
        "best_cv_score": search.best_score_,
        "cv_results": search.cv_results_,
        "tiempo_segundos": duracion,
    }


def entrenar_todos_los_modelos(
    modelos: dict,
    espacios: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict:
    """
    Entrena los 4 modelos con RandomizedSearchCV y devuelve los resultados.

    Parameters
    ----------
    modelos : dict
        Diccionario {nombre: modelo_base}.
    espacios : dict
        Diccionario {nombre: espacio_de_hiperparametros}.
    X_train, y_train : datos de entrenamiento

    Returns
    -------
    dict
        Diccionario {nombre_modelo: dict_de_resultados} (igual estructura
        que entrenar_modelo_con_randomized_search).
    """
    cv = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=SEMILLA)
    logger.info("Validación cruzada: StratifiedKFold con k=%d", N_SPLITS_CV)

    resultados = {}
    tiempo_total = 0.0
    for nombre, modelo in modelos.items():
        resultado = entrenar_modelo_con_randomized_search(
            nombre=nombre,
            modelo=modelo,
            espacio_busqueda=espacios[nombre],
            X_train=X_train,
            y_train=y_train,
            cv=cv,
        )
        resultados[nombre] = resultado
        tiempo_total += resultado["tiempo_segundos"]

    logger.info("-" * 60)
    logger.info(
        "Entrenamiento completado para %d modelos. Tiempo total: %.1f min.",
        len(modelos), tiempo_total / 60
    )

    # Resumen comparativo rápido
    logger.info("Resumen de scores CV (%s):", SCORING_METRIC)
    for nombre, res in sorted(resultados.items(), key=lambda x: -x[1]["best_cv_score"]):
        logger.info("    %-20s | %s = %.4f",
                    nombre, SCORING_METRIC, res["best_cv_score"])

    return resultados


# =========================================================================
# 7. EVALUACIÓN Y COMPARACIÓN
# =========================================================================

def evaluar_modelo(
    nombre: str,
    modelo,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> dict:
    """
    Evalúa un modelo entrenado sobre el conjunto de prueba.

    Calcula un conjunto completo de métricas apropiadas para clasificación
    binaria desbalanceada, prestando atención especial al recall de la
    clase 0 (impago), que es la de interés desde el negocio.

    Parameters
    ----------
    nombre : str
        Nombre del modelo.
    modelo : sklearn estimator
        Modelo entrenado.
    X_test, y_test : datos de prueba
    threshold : float
        Umbral de decisión para convertir probabilidades en clases.

    Returns
    -------
    dict
        Métricas calculadas + probabilidades y predicciones.
    """
    # Probabilidades de la clase positiva (1 = paga a tiempo)
    y_proba = modelo.predict_proba(X_test)[:, 1]
    # Predicciones según el threshold
    y_pred = (y_proba >= threshold).astype(int)

    # Métricas estándar
    metricas = {
        "modelo": nombre,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        # Recall sobre la clase 0 (impago): métrica de negocio.
        "recall_clase_0": recall_score(y_test, y_pred, pos_label=0, zero_division=0),
        "precision_clase_0": precision_score(y_test, y_pred, pos_label=0, zero_division=0),
    }

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    metricas["confusion_matrix"] = cm

    # Guardar probabilidades y predicciones para gráficos posteriores
    metricas["y_proba"] = y_proba
    metricas["y_pred"] = y_pred

    return metricas


def construir_tabla_comparativa(resultados_eval: dict) -> pd.DataFrame:
    """
    Construye una tabla comparativa de las métricas de todos los modelos.

    Parameters
    ----------
    resultados_eval : dict
        Diccionario {nombre_modelo: dict_de_metricas}.

    Returns
    -------
    pd.DataFrame
        Tabla ordenada por ROC-AUC descendente, con las métricas clave.
    """
    filas = []
    for nombre, m in resultados_eval.items():
        filas.append({
            "modelo": nombre,
            "accuracy": round(m["accuracy"], 4),
            "precision": round(m["precision"], 4),
            "recall": round(m["recall"], 4),
            "f1": round(m["f1"], 4),
            "roc_auc": round(m["roc_auc"], 4),
            "pr_auc": round(m["pr_auc"], 4),
            "recall_clase_0": round(m["recall_clase_0"], 4),
            "precision_clase_0": round(m["precision_clase_0"], 4),
        })

    tabla = pd.DataFrame(filas)
    # Ordenar por ROC-AUC desc, desempate por PR-AUC desc
    tabla = tabla.sort_values(
        ["roc_auc", "pr_auc"], ascending=[False, False]
    ).reset_index(drop=True)
    return tabla


def seleccionar_ganador(tabla_comparativa: pd.DataFrame) -> str:
    """
    Selecciona el modelo ganador según ROC-AUC (desempate por PR-AUC).

    Como la tabla ya viene ordenada por (roc_auc desc, pr_auc desc),
    el ganador es simplemente la primera fila.

    Parameters
    ----------
    tabla_comparativa : pd.DataFrame
        Tabla generada por construir_tabla_comparativa.

    Returns
    -------
    str
        Nombre del modelo ganador.
    """
    ganador = tabla_comparativa.iloc[0]["modelo"]
    logger.info("-" * 60)
    logger.info("MODELO GANADOR: %s", ganador)
    logger.info("    ROC-AUC: %.4f | PR-AUC: %.4f | F1: %.4f | Recall clase 0: %.4f",
                tabla_comparativa.iloc[0]["roc_auc"],
                tabla_comparativa.iloc[0]["pr_auc"],
                tabla_comparativa.iloc[0]["f1"],
                tabla_comparativa.iloc[0]["recall_clase_0"])
    return ganador


# =========================================================================
# 8. OPTIMIZACIÓN OPTUNA Y THRESHOLD TUNING
# =========================================================================

def optimizar_logistic_con_optuna(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = N_TRIALS_OPTUNA,
) -> dict:
    """
    Optimiza una Regresión Logística usando Optuna con búsqueda bayesiana.

    A diferencia de RandomizedSearchCV (que usó valores discretos de C),
    Optuna explora C en un rango CONTINUO log-uniforme, lo que permite
    encontrar valores intermedios óptimos.

    Parameters
    ----------
    X_train, y_train : datos de entrenamiento
    n_trials : int
        Número de trials de Optuna.

    Returns
    -------
    dict
        - 'best_estimator': LR entrenada con los mejores hiperparámetros.
        - 'best_params': mejores hiperparámetros.
        - 'best_value': mejor ROC-AUC en CV.
        - 'study': objeto study de Optuna (para graficar historia).
    """
    from sklearn.model_selection import cross_val_score

    cv = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=SEMILLA)

    def objetivo(trial):
        # Espacio de búsqueda continuo
        C = trial.suggest_float("C", 1e-4, 1e3, log=True)
        penalty = trial.suggest_categorical("penalty", ["l1", "l2"])

        modelo = LogisticRegression(
            C=C,
            penalty=penalty,
            solver="liblinear",
            class_weight="balanced",
            max_iter=1000,
            random_state=SEMILLA,
        )
        # ROC-AUC promedio en validación cruzada
        scores = cross_val_score(
            modelo, X_train, y_train,
            scoring=SCORING_METRIC, cv=cv, n_jobs=N_JOBS
        )
        return scores.mean()

    logger.info("-" * 60)
    logger.info("Optimizando Logistic Regression con Optuna (%d trials)...", n_trials)
    t0 = time.time()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEMILLA),
    )
    study.optimize(objetivo, n_trials=n_trials, show_progress_bar=False)

    duracion = time.time() - t0
    logger.info("    Tiempo de optimización: %.1f segundos", duracion)
    logger.info("    Mejor ROC-AUC (CV): %.4f", study.best_value)
    logger.info("    Mejores hiperparámetros: %s", study.best_params)

    # Entrenar el modelo final con los mejores hiperparámetros
    mejor_modelo = LogisticRegression(
        **study.best_params,
        solver="liblinear",
        class_weight="balanced",
        max_iter=1000,
        random_state=SEMILLA,
    )
    mejor_modelo.fit(X_train, y_train)

    return {
        "best_estimator": mejor_modelo,
        "best_params": study.best_params,
        "best_value": study.best_value,
        "study": study,
    }


def optimizar_threshold_por_costo(
    y_test: pd.Series,
    y_proba: np.ndarray,
    costo_impago_no_detectado: int = COSTO_IMPAGO_NO_DETECTADO,
    costo_buen_cliente_rechazado: int = COSTO_BUEN_CLIENTE_RECHAZADO,
) -> dict:
    """
    Encuentra el umbral de decisión que minimiza el costo total de negocio.

    Definición de costos (clase positiva = 1 = paga a tiempo):

        - Impago no detectado (real=0, predicho=1): el cliente NO paga pero
          el modelo lo aprueba como buen pagador. Es el error más costoso
          (se otorga crédito que no se recuperará). En la matriz de
          confusión con labels [0, 1] esto corresponde a FP (falso positivo
          de la clase 1).

        - Buen cliente rechazado (real=1, predicho=0): el cliente SÍ paga
          pero el modelo lo marca como riesgo. Implica perder un buen
          negocio (costo de oportunidad), pero es menos grave. Corresponde
          a FN (falso negativo de la clase 1).

    El ratio de costos (por defecto 20:1) refleja que dejar pasar un impago
    es ~20 veces más costoso que rechazar un buen cliente, proporción
    alineada con el desbalance del dataset (~20:1).

    Parameters
    ----------
    y_test : pd.Series
        Etiquetas verdaderas.
    y_proba : np.ndarray
        Probabilidades de la clase 1.
    costo_impago_no_detectado : int
        Costo de aprobar a quien no paga (evento grave).
    costo_buen_cliente_rechazado : int
        Costo de rechazar a quien sí paga.

    Returns
    -------
    dict
        - 'threshold_optimo': umbral que minimiza el costo.
        - 'costo_minimo': costo total en ese umbral.
        - 'tabla_costos': DataFrame con threshold, costo y métricas.
    """
    thresholds = np.arange(0.01, 1.00, 0.01)
    registros = []

    for th in thresholds:
        y_pred = (y_proba >= th).astype(int)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        # cm con labels [0, 1]:
        #   [[TN, FP],
        #    [FN, TP]]
        # donde:
        #   TN = real 0, pred 0  -> impago detectado correctamente
        #   FP = real 0, pred 1  -> IMPAGO NO DETECTADO (grave)
        #   FN = real 1, pred 0  -> BUEN CLIENTE RECHAZADO
        #   TP = real 1, pred 1  -> buen cliente aprobado correctamente
        tn, fp, fn, tp = cm.ravel()

        costo_total = (
            costo_impago_no_detectado * fp +
            costo_buen_cliente_rechazado * fn
        )

        registros.append({
            "threshold": round(th, 2),
            "costo_total": int(costo_total),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "recall_clase_0": round(recall_score(y_test, y_pred, pos_label=0, zero_division=0), 4),
        })

    tabla_costos = pd.DataFrame(registros)
    idx_min = tabla_costos["costo_total"].idxmin()
    threshold_optimo = float(tabla_costos.loc[idx_min, "threshold"])
    costo_minimo = int(tabla_costos.loc[idx_min, "costo_total"])

    logger.info("-" * 60)
    logger.info(
        "Threshold tuning (costo impago_no_detectado=%d, buen_cliente_rechazado=%d):",
        costo_impago_no_detectado, costo_buen_cliente_rechazado
    )
    logger.info("    Threshold óptimo: %.2f", threshold_optimo)
    logger.info("    Costo mínimo: %d", costo_minimo)
    fila_opt = tabla_costos.loc[idx_min]
    logger.info("    Métricas en threshold óptimo:")
    logger.info("        precision=%.3f recall=%.3f f1=%.3f recall_clase_0=%.3f",
                fila_opt["precision"], fila_opt["recall"],
                fila_opt["f1"], fila_opt["recall_clase_0"])
    logger.info("        Impagos no detectados (FP)=%d | Buenos rechazados (FN)=%d",
                int(fila_opt["FP"]), int(fila_opt["FN"]))

    # Comparar con threshold default 0.5
    fila_default = tabla_costos[tabla_costos["threshold"] == 0.50]
    if len(fila_default) > 0:
        costo_default = int(fila_default.iloc[0]["costo_total"])
        rec0_default = float(fila_default.iloc[0]["recall_clase_0"])
        logger.info("    Comparación con threshold 0.50: costo=%d, recall_clase_0=%.3f",
                    costo_default, rec0_default)

    return {
        "threshold_optimo": threshold_optimo,
        "costo_minimo": costo_minimo,
        "tabla_costos": tabla_costos,
    }


# =========================================================================
# 9. PERSISTENCIA Y REPORTES VISUALES
# =========================================================================

def guardar_artefactos_modelo(
    modelo_final,
    nombre_modelo: str,
    threshold_optimo: float,
    eval_final: dict,
    tabla_comparativa: pd.DataFrame,
    best_params: dict,
) -> None:
    """
    Persiste el modelo ganador, el threshold óptimo y las métricas finales.

    Genera 3 archivos en artifacts/models/:
        - modelo_ganador.pkl  (modelo serializado con joblib)
        - threshold_optimo.json
        - metricas_finales.json

    Parameters
    ----------
    modelo_final : sklearn estimator
        Modelo final (optimizado por Optuna).
    nombre_modelo : str
        Nombre del modelo ganador.
    threshold_optimo : float
        Umbral de decisión óptimo.
    eval_final : dict
        Métricas del modelo final (con threshold 0.5).
    tabla_comparativa : pd.DataFrame
        Tabla comparativa de todos los modelos.
    best_params : dict
        Hiperparámetros del modelo final.
    """
    # 1. Modelo serializado
    ruta_modelo = RUTA_MODELS / "modelo_ganador.pkl"
    joblib.dump(modelo_final, ruta_modelo)
    logger.info("Modelo serializado: %s", ruta_modelo.relative_to(RUTA_RAIZ))

    # 2. Threshold óptimo
    ruta_threshold = RUTA_MODELS / "threshold_optimo.json"
    threshold_data = {
        "threshold_optimo": threshold_optimo,
        "threshold_default": 0.5,
        "costo_impago_no_detectado": COSTO_IMPAGO_NO_DETECTADO,
        "costo_buen_cliente_rechazado": COSTO_BUEN_CLIENTE_RECHAZADO,
    }
    with open(ruta_threshold, "w", encoding="utf-8") as f:
        json.dump(threshold_data, f, ensure_ascii=False, indent=2)
    logger.info("Threshold guardado: %s", ruta_threshold.relative_to(RUTA_RAIZ))

    # 3. Métricas finales
    ruta_metricas = RUTA_MODELS / "metricas_finales.json"
    metricas_data = {
        "modelo_ganador": nombre_modelo,
        "hiperparametros": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                             for k, v in best_params.items()},
        "threshold_optimo": threshold_optimo,
        "metricas_test_threshold_default": {
            "accuracy": round(float(eval_final["accuracy"]), 4),
            "precision": round(float(eval_final["precision"]), 4),
            "recall": round(float(eval_final["recall"]), 4),
            "f1": round(float(eval_final["f1"]), 4),
            "roc_auc": round(float(eval_final["roc_auc"]), 4),
            "pr_auc": round(float(eval_final["pr_auc"]), 4),
            "recall_clase_0": round(float(eval_final["recall_clase_0"]), 4),
            "precision_clase_0": round(float(eval_final["precision_clase_0"]), 4),
        },
        "tabla_comparativa": tabla_comparativa.drop(
            columns=[c for c in tabla_comparativa.columns if c in []],
            errors="ignore"
        ).to_dict(orient="records"),
    }
    with open(ruta_metricas, "w", encoding="utf-8") as f:
        json.dump(metricas_data, f, ensure_ascii=False, indent=2)
    logger.info("Métricas guardadas: %s", ruta_metricas.relative_to(RUTA_RAIZ))


def guardar_tabla_comparativa(tabla: pd.DataFrame) -> None:
    """Guarda la tabla comparativa como CSV y como imagen PNG."""
    # CSV
    ruta_csv = RUTA_REPORTS / "tabla_comparativa_modelos.csv"
    tabla.to_csv(ruta_csv, index=False, encoding="utf-8")
    logger.info("Tabla comparativa (CSV): %s", ruta_csv.relative_to(RUTA_RAIZ))

    # PNG (render de la tabla)
    fig, ax = plt.subplots(figsize=(13, 2 + 0.4 * len(tabla)))
    ax.axis("off")
    tabla_render = ax.table(
        cellText=tabla.values,
        colLabels=tabla.columns,
        cellLoc="center",
        loc="center",
    )
    tabla_render.auto_set_font_size(False)
    tabla_render.set_fontsize(9)
    tabla_render.scale(1.2, 1.5)
    # Resaltar la fila del ganador (primera)
    for j in range(len(tabla.columns)):
        tabla_render[(1, j)].set_facecolor("#D6EAF8")
    ax.set_title("Comparación de modelos (ordenados por ROC-AUC)", fontsize=12, pad=20)
    ruta_png = RUTA_REPORTS / "tabla_comparativa_modelos.png"
    plt.savefig(ruta_png, dpi=100, bbox_inches="tight")
    plt.close(fig)
    logger.info("Tabla comparativa (PNG): %s", ruta_png.relative_to(RUTA_RAIZ))


def graficar_roc_comparativa(resultados_eval: dict, y_test: pd.Series) -> None:
    """Grafica las curvas ROC de todos los modelos en un solo plot."""
    fig, ax = plt.subplots(figsize=(8, 7))
    for nombre, m in resultados_eval.items():
        fpr, tpr, _ = roc_curve(y_test, m["y_proba"])
        ax.plot(fpr, tpr, linewidth=2,
                label=f"{nombre} (AUC={m['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Azar (AUC=0.5)")
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title("Curvas ROC comparativas")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    ruta = RUTA_REPORTS / "roc_curve_comparativa.png"
    plt.savefig(ruta, dpi=100, bbox_inches="tight")
    plt.close(fig)
    logger.info("Reporte visual: %s", ruta.relative_to(RUTA_RAIZ))


def graficar_pr_comparativa(resultados_eval: dict, y_test: pd.Series) -> None:
    """Grafica las curvas Precision-Recall de todos los modelos."""
    fig, ax = plt.subplots(figsize=(8, 7))
    for nombre, m in resultados_eval.items():
        precision, recall, _ = precision_recall_curve(y_test, m["y_proba"])
        ax.plot(recall, precision, linewidth=2,
                label=f"{nombre} (PR-AUC={m['pr_auc']:.3f})")
    # Línea base = proporción de la clase positiva
    baseline = y_test.mean()
    ax.axhline(y=baseline, color="k", linestyle="--", linewidth=1,
               label=f"Base (={baseline:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curvas Precision-Recall comparativas")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    ruta = RUTA_REPORTS / "precision_recall_curve_comparativa.png"
    plt.savefig(ruta, dpi=100, bbox_inches="tight")
    plt.close(fig)
    logger.info("Reporte visual: %s", ruta.relative_to(RUTA_RAIZ))


def graficar_matriz_confusion(
    y_test: pd.Series,
    y_proba: np.ndarray,
    threshold: float,
    titulo_sufijo: str,
    nombre_archivo: str,
) -> None:
    """Grafica una matriz de confusión para un threshold dado."""
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["No paga (0)", "Paga (1)"],
        yticklabels=["No paga (0)", "Paga (1)"],
        ax=ax,
    )
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusión {titulo_sufijo}\n(threshold={threshold:.2f})")
    ruta = RUTA_REPORTS / nombre_archivo
    plt.savefig(ruta, dpi=100, bbox_inches="tight")
    plt.close(fig)
    logger.info("Reporte visual: %s", ruta.relative_to(RUTA_RAIZ))


def graficar_feature_importance(modelo, feature_names: list, nombre_modelo: str) -> None:
    """
    Grafica la importancia de features del modelo ganador.

    Soporta modelos con feature_importances_ (árboles) y con coef_ (lineales).
    """
    if hasattr(modelo, "feature_importances_"):
        importancias = modelo.feature_importances_
        tipo = "feature_importances_"
    elif hasattr(modelo, "coef_"):
        # Para modelos lineales, usar el valor absoluto de los coeficientes
        importancias = np.abs(modelo.coef_).ravel()
        tipo = "|coeficientes|"
    else:
        logger.warning("El modelo no tiene feature_importances_ ni coef_; se omite el gráfico.")
        return

    serie = pd.Series(importancias, index=feature_names).sort_values(ascending=True)
    # Mostrar top 20 para legibilidad
    serie = serie.tail(20)

    fig, ax = plt.subplots(figsize=(10, 8))
    serie.plot(kind="barh", ax=ax, color="steelblue", edgecolor="white")
    ax.set_xlabel(f"Importancia ({tipo})")
    ax.set_ylabel("Feature")
    ax.set_title(f"Importancia de features - {nombre_modelo} (top 20)")
    plt.tight_layout()
    ruta = RUTA_REPORTS / "feature_importance.png"
    plt.savefig(ruta, dpi=100, bbox_inches="tight")
    plt.close(fig)
    logger.info("Reporte visual: %s", ruta.relative_to(RUTA_RAIZ))


def graficar_curva_costo(tabla_costos: pd.DataFrame, threshold_optimo: float) -> None:
    """Grafica la curva de costo total vs threshold, marcando el óptimo."""
    fig, ax1 = plt.subplots(figsize=(11, 6))

    # Eje izquierdo: costo total
    ax1.plot(tabla_costos["threshold"], tabla_costos["costo_total"],
             color="#D9534F", linewidth=2, label="Costo total")
    ax1.axvline(x=threshold_optimo, color="green", linestyle="--", linewidth=1.5,
                label=f"Threshold óptimo ({threshold_optimo:.2f})")
    ax1.axvline(x=0.5, color="gray", linestyle=":", linewidth=1.5,
                label="Threshold default (0.50)")
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Costo total", color="#D9534F")
    ax1.tick_params(axis="y", labelcolor="#D9534F")
    ax1.grid(alpha=0.3)

    # Eje derecho: recall clase 0
    ax2 = ax1.twinx()
    ax2.plot(tabla_costos["threshold"], tabla_costos["recall_clase_0"],
             color="#5BC0DE", linewidth=1.5, alpha=0.7, label="Recall clase 0")
    ax2.set_ylabel("Recall clase 0 (detección de impagos)", color="#5BC0DE")
    ax2.tick_params(axis="y", labelcolor="#5BC0DE")

    # Combinar leyendas
    lineas1, labels1 = ax1.get_legend_handles_labels()
    lineas2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lineas1 + lineas2, labels1 + labels2, loc="upper center")

    ax1.set_title("Costo total vs Threshold (con recall de clase 0)")
    plt.tight_layout()
    ruta = RUTA_REPORTS / "curva_costo_vs_threshold.png"
    plt.savefig(ruta, dpi=100, bbox_inches="tight")
    plt.close(fig)
    logger.info("Reporte visual: %s", ruta.relative_to(RUTA_RAIZ))


def graficar_optuna_history(study, nombre_modelo: str) -> None:
    """Grafica la evolución del ROC-AUC a lo largo de los trials de Optuna."""
    if study is None:
        logger.info("No hay estudio de Optuna que graficar (modelo no-LR).")
        return

    valores = [t.value for t in study.trials if t.value is not None]
    mejores_acumulados = np.maximum.accumulate(valores)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(valores) + 1), valores, "o-", alpha=0.4,
            color="steelblue", label="ROC-AUC por trial")
    ax.plot(range(1, len(mejores_acumulados) + 1), mejores_acumulados,
            color="green", linewidth=2, label="Mejor acumulado")
    ax.set_xlabel("Trial")
    ax.set_ylabel("ROC-AUC (CV)")
    ax.set_title(f"Historial de optimización Optuna - {nombre_modelo}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    ruta = RUTA_REPORTS / "optuna_history.png"
    plt.savefig(ruta, dpi=100, bbox_inches="tight")
    plt.close(fig)
    logger.info("Reporte visual: %s", ruta.relative_to(RUTA_RAIZ))


# =========================================================================
# 10. FUNCIÓN PRINCIPAL
# =========================================================================
def main() -> None:
    """
    Orquesta el flujo completo de entrenamiento y evaluación de modelos.

    Pasos (a desarrollar en las siguientes etapas):
        1. Cargar datasets transformados y pipeline.
        2. Definir modelos base con manejo de desbalance.
        3. Entrenar los 4 modelos con RandomizedSearchCV.
        4. Evaluar y comparar.
        5. Seleccionar el ganador.
        6. Optimizar con Optuna.
        7. Threshold tuning.
        8. Persistir modelo + reportes.
    """
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("Inicio del pipeline de entrenamiento y evaluación (V1.0.1)")
    logger.info("=" * 60)

    # Verificación: ¿existen los artefactos del feature engineering?
    archivos_esperados = [
        RUTA_DATA / "X_train.parquet",
        RUTA_DATA / "X_test.parquet",
        RUTA_DATA / "y_train.parquet",
        RUTA_DATA / "y_test.parquet",
    ]
    faltantes = [str(p.relative_to(RUTA_RAIZ)) for p in archivos_esperados if not p.exists()]
    if faltantes:
        logger.error(
            "Artefactos faltantes (debe ejecutar ft_engineering.py primero): %s",
            faltantes
        )
        raise FileNotFoundError(
            "Faltan artefactos del feature engineering. "
            "Ejecuta primero 'python mlops_pipeline/src/ft_engineering.py'."
        )

    # Verificación: ¿existen las carpetas de destino?
    for carpeta in [RUTA_MODELS, RUTA_REPORTS]:
        carpeta.mkdir(parents=True, exist_ok=True)
        logger.info("Carpeta verificada: %s", carpeta.relative_to(RUTA_RAIZ))

    # -----------------------------------------------------------------
    # Paso 1: Cargar artefactos del feature engineering
    # -----------------------------------------------------------------
    X_train, X_test, y_train, y_test = cargar_datasets_transformados()
    # Nota: el pipeline serializado NO se carga aquí. Se cargará en el
    # script de despliegue (Avance #4) donde sí estará disponible el
    # módulo ft_engineering para reconstruir los Transformers personalizados.

    # -----------------------------------------------------------------
    # Paso 2: Definir modelos base y espacios de búsqueda
    # -----------------------------------------------------------------
    modelos = obtener_modelos_base(y_train)
    espacios = obtener_espacios_busqueda()

    logger.info("Modelos base definidos: %d", len(modelos))
    for nombre, modelo in modelos.items():
        n_params_busqueda = len(espacios[nombre])
        logger.info(
            "    %-20s | %d hiperparámetros a optimizar",
            nombre, n_params_busqueda
        )

    # -----------------------------------------------------------------
    # Paso 3: Entrenar los 4 modelos con RandomizedSearchCV
    # -----------------------------------------------------------------
    resultados_entrenamiento = entrenar_todos_los_modelos(
        modelos=modelos,
        espacios=espacios,
        X_train=X_train,
        y_train=y_train,
    )

    # -----------------------------------------------------------------
    # Paso 4: Evaluar cada modelo sobre el test set
    # -----------------------------------------------------------------
    logger.info("-" * 60)
    logger.info("Evaluando modelos sobre el conjunto de prueba...")
    resultados_eval = {}
    for nombre, res in resultados_entrenamiento.items():
        resultados_eval[nombre] = evaluar_modelo(
            nombre=nombre,
            modelo=res["best_estimator"],
            X_test=X_test,
            y_test=y_test,
        )

    # -----------------------------------------------------------------
    # Paso 5: Tabla comparativa y selección del ganador
    # -----------------------------------------------------------------
    tabla_comparativa = construir_tabla_comparativa(resultados_eval)
    logger.info("-" * 60)
    logger.info("Tabla comparativa de modelos (ordenada por ROC-AUC):")
    # Imprimir la tabla línea por línea para el log
    for _, fila in tabla_comparativa.iterrows():
        logger.info(
            "    %-20s | acc=%.3f prec=%.3f rec=%.3f f1=%.3f roc=%.4f pr=%.4f rec0=%.3f",
            fila["modelo"], fila["accuracy"], fila["precision"], fila["recall"],
            fila["f1"], fila["roc_auc"], fila["pr_auc"], fila["recall_clase_0"]
        )

    nombre_ganador = seleccionar_ganador(tabla_comparativa)
    modelo_ganador = resultados_entrenamiento[nombre_ganador]["best_estimator"]

    # -----------------------------------------------------------------
    # Paso 6: Optimización con Optuna sobre el ganador
    # -----------------------------------------------------------------
    # El ganador esperado es Logistic Regression. Optuna explora C continuo.
    # (Si el ganador fuera otro modelo, este paso requeriría adaptación;
    # por ahora la optimización Optuna está implementada para LR.)
    if nombre_ganador == "LogisticRegression":
        resultado_optuna = optimizar_logistic_con_optuna(X_train, y_train)
        modelo_optimizado = resultado_optuna["best_estimator"]
        study_optuna = resultado_optuna["study"]

        # Evaluar el modelo optimizado por Optuna
        eval_optimizado = evaluar_modelo(
            nombre=f"{nombre_ganador}_optuna",
            modelo=modelo_optimizado,
            X_test=X_test,
            y_test=y_test,
        )
        logger.info("Modelo optimizado por Optuna - ROC-AUC test: %.4f (vs %.4f del RandomizedSearch)",
                    eval_optimizado["roc_auc"],
                    resultados_eval[nombre_ganador]["roc_auc"])

        # Usar el modelo optimizado como modelo final
        modelo_final = modelo_optimizado
        eval_final = eval_optimizado
    else:
        logger.warning(
            "El ganador (%s) no es LogisticRegression. "
            "Se usa el modelo de RandomizedSearch sin optimización Optuna adicional.",
            nombre_ganador
        )
        modelo_final = modelo_ganador
        eval_final = resultados_eval[nombre_ganador]
        study_optuna = None

    # -----------------------------------------------------------------
    # Paso 7: Threshold tuning sobre el modelo final
    # -----------------------------------------------------------------
    resultado_threshold = optimizar_threshold_por_costo(
        y_test=y_test,
        y_proba=eval_final["y_proba"],
    )
    threshold_optimo = resultado_threshold["threshold_optimo"]

    # -----------------------------------------------------------------
    # Paso 8: Persistencia de artefactos del modelo
    # -----------------------------------------------------------------
    logger.info("-" * 60)
    logger.info("Persistiendo artefactos del modelo...")

    # Hiperparámetros del modelo final
    if study_optuna is not None:
        best_params = resultado_optuna["best_params"]
    else:
        best_params = resultados_entrenamiento[nombre_ganador]["best_params"]

    guardar_artefactos_modelo(
        modelo_final=modelo_final,
        nombre_modelo=nombre_ganador,
        threshold_optimo=threshold_optimo,
        eval_final=eval_final,
        tabla_comparativa=tabla_comparativa,
        best_params=best_params,
    )

    # -----------------------------------------------------------------
    # Paso 9: Reportes visuales
    # -----------------------------------------------------------------
    logger.info("-" * 60)
    logger.info("Generando reportes visuales...")

    guardar_tabla_comparativa(tabla_comparativa)
    graficar_roc_comparativa(resultados_eval, y_test)
    graficar_pr_comparativa(resultados_eval, y_test)
    graficar_matriz_confusion(
        y_test, eval_final["y_proba"], threshold=0.5,
        titulo_sufijo="(threshold default)",
        nombre_archivo="matriz_confusion_ganador_threshold_default.png",
    )
    graficar_matriz_confusion(
        y_test, eval_final["y_proba"], threshold=threshold_optimo,
        titulo_sufijo="(threshold optimizado)",
        nombre_archivo="matriz_confusion_ganador_threshold_optimizado.png",
    )
    graficar_feature_importance(modelo_final, X_train.columns.tolist(), nombre_ganador)
    graficar_curva_costo(resultado_threshold["tabla_costos"], threshold_optimo)
    graficar_optuna_history(study_optuna, nombre_ganador)

    logger.info("-" * 60)
    logger.info("Pipeline de entrenamiento V1.0.1 ejecutado completamente.")

    duracion = time.time() - t0
    logger.info("=" * 60)
    logger.info("Fin del pipeline. Tiempo total: %.1f segundos.", duracion)
    logger.info("=" * 60)


# =========================================================================
# 11. PUNTO DE ENTRADA
# =========================================================================
if __name__ == "__main__":
    main()
