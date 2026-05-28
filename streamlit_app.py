"""
streamlit_app.py
Dashboard interactivo para el modelo de riesgo crediticio (PIM5).

Aplicación web desarrollada con Streamlit que ofrece tres secciones:
    1. Predicción interactiva: ingreso de datos de un cliente y predicción
       de la probabilidad de pago a tiempo.
    2. Monitoreo de drift: visualización de los reportes de data drift
       generados por model_monitoring.py.
    3. Métricas del modelo: desempeño del modelo seleccionado y comparación
       entre los modelos entrenados.

Ejecución:
    streamlit run streamlit_app.py

Requisitos previos:
    Haber ejecutado ft_engineering.py, model_training_evaluation.py y
    model_monitoring.py para disponer de los artefactos necesarios.

Autor: Cristian García
Fecha de creación: 2026-05-28
Versión: V1.4.0
"""

# =========================================================================
# 1. IMPORTS Y CONFIGURACIÓN DE RUTAS
# =========================================================================
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Para poder cargar el pipeline serializado (que contiene los Transformers
# personalizados), Python debe poder importar las definiciones de esas
# clases desde ft_engineering.py. Agregamos la carpeta src al path.
RUTA_RAIZ = Path(__file__).resolve().parent
RUTA_SRC = RUTA_RAIZ / "mlops_pipeline" / "src"
if str(RUTA_SRC) not in sys.path:
    sys.path.insert(0, str(RUTA_SRC))

# Importar los Transformers personalizados para que joblib pueda
# reconstruir el pipeline. (Solución 1 al acoplamiento.)
try:
    from ft_engineering import (
        NullIndicatorTransformer,
        Log1pTransformer,
        limpiar_tendencia_ingresos,
        aplicar_capping_anomalias,
        convertir_categoricas_codificadas_a_string,
    )  # noqa: F401
    FT_ENGINEERING_OK = True
except ImportError:
    # Si falla la importación, se mostrará un mensaje en la app.
    NullIndicatorTransformer = None
    Log1pTransformer = None
    FT_ENGINEERING_OK = False

# El pipeline (pipeline_ft_engineering.pkl) se serializó mientras
# ft_engineering.py se ejecutaba como script principal, por lo que las
# clases de los Transformers quedaron referenciadas bajo el módulo
# '__main__'. Para que joblib pueda reconstruir el pipeline desde este
# dashboard, registramos esas clases en __main__ explícitamente.
# (Esta es una técnica estándar para cargar pickles generados desde scripts.)
if FT_ENGINEERING_OK:
    import __main__
    __main__.NullIndicatorTransformer = NullIndicatorTransformer
    __main__.Log1pTransformer = Log1pTransformer

    # Silenciar los logs informativos de ft_engineering (limpieza, capping)
    # para que no aparezcan en la terminal en cada predicción.
    import logging
    logging.getLogger("ft_engineering").setLevel(logging.WARNING)


# =========================================================================
# 2. CONSTANTES DE RUTAS A ARTEFACTOS
# =========================================================================
RUTA_ARTIFACTS = RUTA_RAIZ / "artifacts"
RUTA_DATA = RUTA_ARTIFACTS / "data"
RUTA_TRANSFORMERS = RUTA_ARTIFACTS / "transformers"
RUTA_MODELS = RUTA_ARTIFACTS / "models"
RUTA_MONITORING = RUTA_ARTIFACTS / "monitoring"

RUTA_REPORTS_MODEL = RUTA_RAIZ / "reports" / "model_training"
RUTA_REPORTS_MONITORING = RUTA_RAIZ / "reports" / "monitoring"

RUTA_DATASET = RUTA_RAIZ / "Base_de_datos.xlsx"

COL_TARGET = "Pago_atiempo"


# =========================================================================
# 3. FUNCIONES DE CARGA (CACHEADAS)
# =========================================================================

@st.cache_resource
def cargar_modelo():
    """Carga el modelo ganador serializado. Cacheado entre interacciones."""
    ruta = RUTA_MODELS / "modelo_ganador.pkl"
    if not ruta.exists():
        return None
    return joblib.load(ruta)


@st.cache_resource
def cargar_pipeline():
    """Carga el pipeline de feature engineering. Cacheado."""
    ruta = RUTA_TRANSFORMERS / "pipeline_ft_engineering.pkl"
    if not ruta.exists():
        return None
    return joblib.load(ruta)


@st.cache_data
def cargar_metricas():
    """Carga las métricas finales del modelo (JSON). Cacheado."""
    ruta = RUTA_MODELS / "metricas_finales.json"
    if not ruta.exists():
        return None
    with open(ruta, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def cargar_threshold():
    """Carga el threshold óptimo (JSON). Cacheado."""
    ruta = RUTA_MODELS / "threshold_optimo.json"
    if not ruta.exists():
        return None
    with open(ruta, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def cargar_psi_report():
    """Carga el reporte de PSI (CSV). Cacheado."""
    ruta = RUTA_REPORTS_MONITORING / "psi_report.csv"
    if not ruta.exists():
        return None
    return pd.read_csv(ruta)


@st.cache_data
def cargar_dataset_crudo():
    """Carga el dataset crudo para calcular valores por defecto. Cacheado."""
    if not RUTA_DATASET.exists():
        return None
    return pd.read_excel(RUTA_DATASET)


@st.cache_data
def obtener_valores_por_defecto():
    """
    Calcula valores por defecto (mediana para numéricas, moda para
    categóricas) del dataset crudo, para rellenar las features que el
    usuario no ingresa manualmente.

    Returns
    -------
    dict
        {nombre_columna: valor_por_defecto} para todas las columnas
        excepto el target.
    """
    df = cargar_dataset_crudo()
    if df is None:
        return {}

    defaults = {}
    for col in df.columns:
        if col == COL_TARGET:
            continue
        serie = df[col]
        if pd.api.types.is_numeric_dtype(serie):
            defaults[col] = float(serie.median())
        else:
            moda = serie.mode()
            defaults[col] = moda.iloc[0] if len(moda) > 0 else "DESCONOCIDO"
    return defaults


def construir_registro_cliente(inputs_usuario: dict) -> pd.DataFrame:
    """
    Construye un DataFrame de una fila con todas las columnas que el
    pipeline espera, combinando los inputs del usuario con los valores
    por defecto para las columnas no ingresadas.

    Luego aplica la misma limpieza que ft_engineering.py para que el
    registro sea compatible con el pipeline.

    Parameters
    ----------
    inputs_usuario : dict
        Valores ingresados por el usuario (8 campos clave).

    Returns
    -------
    pd.DataFrame
        DataFrame de una fila, limpio y listo para pipeline.transform().
    """
    defaults = obtener_valores_por_defecto()

    # Combinar: empezar con defaults y sobrescribir con inputs del usuario
    registro = dict(defaults)
    registro.update(inputs_usuario)

    # Quitar el target si quedó en los defaults
    registro.pop(COL_TARGET, None)

    df_registro = pd.DataFrame([registro])

    # Aplicar la misma limpieza que ft_engineering.py
    df_registro = limpiar_tendencia_ingresos(df_registro)
    df_registro = aplicar_capping_anomalias(df_registro)
    df_registro = convertir_categoricas_codificadas_a_string(df_registro)

    # Quitar columnas que el pipeline descarta (fecha y leakage).
    # El pipeline fue entrenado sobre X que ya no tenía estas columnas.
    for col in ["fecha_prestamo", "puntaje"]:
        if col in df_registro.columns:
            df_registro = df_registro.drop(columns=[col])

    return df_registro


# =========================================================================
# 4. CONFIGURACIÓN DE LA PÁGINA
# =========================================================================
st.set_page_config(
    page_title="Riesgo Crediticio - PIM5",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================================
# 5. ENCABEZADO
# =========================================================================
st.title("Modelo de Riesgo Crediticio")
st.caption(
    "Dashboard interactivo - Proyecto Integrador Módulo 5 - Ciencia de Datos"
)

# Verificación de artefactos mínimos
modelo = cargar_modelo()
pipeline = cargar_pipeline()

if modelo is None or pipeline is None:
    st.error(
        "No se encontraron los artefactos del modelo. "
        "Ejecuta primero los scripts ft_engineering.py y "
        "model_training_evaluation.py desde la raíz del proyecto."
    )
    st.stop()


# =========================================================================
# 6. ESTRUCTURA DE PESTAÑAS
# =========================================================================
tab_prediccion, tab_drift, tab_metricas = st.tabs([
    "Predicción de cliente",
    "Monitoreo de drift",
    "Métricas del modelo",
])

with tab_prediccion:
    st.header("Predicción de riesgo para un cliente")
    st.markdown(
        "Ingresa los datos del cliente. Los campos no mostrados se completan "
        "automáticamente con valores típicos de la cartera."
    )

    threshold_data = cargar_threshold()
    threshold_optimo = threshold_data["threshold_optimo"] if threshold_data else 0.5

    # Cargar dataset para definir rangos razonables de los sliders
    df_crudo = cargar_dataset_crudo()

    col1, col2 = st.columns(2)

    with col1:
        edad = st.slider("Edad del cliente", 18, 100, 40)
        salario = st.number_input(
            "Salario mensual (COP)", min_value=0, max_value=100_000_000,
            value=3_000_000, step=100_000
        )
        capital = st.number_input(
            "Capital prestado (COP)", min_value=0, max_value=50_000_000,
            value=2_000_000, step=100_000
        )
        plazo = st.slider("Plazo (meses)", 1, 72, 24)

    with col2:
        puntaje_dc = st.slider("Puntaje Datacrédito", 1, 950, 600)
        tipo_laboral = st.selectbox(
            "Tipo laboral", ["Empleado", "Independiente"]
        )
        tendencia = st.selectbox(
            "Tendencia de ingresos",
            ["Creciente", "Estable", "Decreciente", "DESCONOCIDO"]
        )
        # Tipos de crédito disponibles en el dataset (codificados)
        if df_crudo is not None and "tipo_credito" in df_crudo.columns:
            tipos_credito = sorted(df_crudo["tipo_credito"].dropna().unique().tolist())
        else:
            tipos_credito = [1, 2, 3, 4, 5, 6]
        tipo_credito = st.selectbox("Tipo de crédito", tipos_credito)

    if st.button("Predecir", type="primary"):
        inputs_usuario = {
            "edad_cliente": edad,
            "salario_cliente": salario,
            "capital_prestado": capital,
            "plazo_meses": plazo,
            "puntaje_datacredito": puntaje_dc,
            "tipo_laboral": tipo_laboral,
            "tendencia_ingresos": tendencia,
            "tipo_credito": tipo_credito,
        }

        try:
            # Construir registro completo y transformar
            df_registro = construir_registro_cliente(inputs_usuario)
            X_transformado = pipeline.transform(df_registro)

            # Predecir probabilidad de la clase 1 (paga a tiempo)
            proba_paga = float(modelo.predict_proba(X_transformado)[:, 1][0])

            # Aplicar threshold óptimo
            aprueba = proba_paga >= threshold_optimo

            st.markdown("---")
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.metric(
                    "Probabilidad de pago a tiempo",
                    f"{proba_paga * 100:.1f}%"
                )
                st.progress(proba_paga)
            with res_col2:
                if aprueba:
                    st.success(
                        f"DECISIÓN: APROBAR\n\n"
                        f"La probabilidad ({proba_paga * 100:.1f}%) supera el "
                        f"umbral óptimo ({threshold_optimo * 100:.0f}%)."
                    )
                else:
                    st.warning(
                        f"DECISIÓN: RECHAZAR / REVISAR\n\n"
                        f"La probabilidad ({proba_paga * 100:.1f}%) está por "
                        f"debajo del umbral óptimo ({threshold_optimo * 100:.0f}%)."
                    )

            st.caption(
                "Nota: el umbral óptimo se calculó minimizando el costo de "
                "negocio (no detectar un impago es ~20 veces más costoso que "
                "rechazar un buen cliente)."
            )
        except Exception as e:
            st.error(f"Error al generar la predicción: {e}")

with tab_drift:
    st.header("Monitoreo de data drift")
    st.markdown(
        "El **data drift** ocurre cuando la distribución de los datos nuevos "
        "difiere de la distribución con la que se entrenó el modelo. Se mide "
        "con el **PSI (Population Stability Index)**:"
    )
    st.markdown(
        "- **PSI < 0.10**: sin drift significativo (estable).\n"
        "- **0.10 <= PSI < 0.25**: drift moderado (vigilar).\n"
        "- **PSI >= 0.25**: drift severo (considerar reentrenamiento)."
    )

    psi_report = cargar_psi_report()

    if psi_report is None:
        st.warning(
            "No se encontró el reporte de drift. Ejecuta primero "
            "model_monitoring.py para generar los reportes."
        )
    else:
        # Resumen tipo semáforo por escenario
        st.subheader("Resumen por escenario")
        escenarios = psi_report["escenario"].unique()
        cols_resumen = st.columns(len(escenarios))

        for col, esc in zip(cols_resumen, escenarios):
            sub = psi_report[psi_report["escenario"] == esc]
            n_sin = (sub["psi_interpretacion"] == "sin_drift").sum()
            n_mod = (sub["psi_interpretacion"] == "drift_moderado").sum()
            n_sev = (sub["psi_interpretacion"] == "drift_severo").sum()
            with col:
                etiqueta = "Sin drift" if esc == "sin_drift" else "Con drift inducido"
                st.markdown(f"**{etiqueta}**")
                st.metric("Variables estables", n_sin)
                st.metric("Drift moderado", n_mod)
                st.metric("Drift severo", n_sev)

        # Imagen del resumen de PSI
        ruta_summary = RUTA_REPORTS_MONITORING / "drift_summary.png"
        if ruta_summary.exists():
            st.subheader("PSI por variable (ambos escenarios)")
            st.image(str(ruta_summary), use_container_width=True)

        # Tabla interactiva filtrable
        st.subheader("Detalle por variable")
        escenario_sel = st.radio(
            "Escenario a mostrar:",
            options=list(escenarios),
            format_func=lambda x: "Sin drift" if x == "sin_drift" else "Con drift inducido",
            horizontal=True,
        )
        tabla_filtrada = psi_report[psi_report["escenario"] == escenario_sel].copy()
        tabla_filtrada = tabla_filtrada.sort_values("psi", ascending=False)
        st.dataframe(
            tabla_filtrada[["variable", "tipo", "psi", "psi_interpretacion"]],
            use_container_width=True,
            hide_index=True,
        )

        # Imagen de distribuciones
        ruta_dist = RUTA_REPORTS_MONITORING / "distribuciones_drift.png"
        if ruta_dist.exists():
            st.subheader("Distribuciones de las variables con mayor drift")
            st.image(str(ruta_dist), use_container_width=True)

with tab_metricas:
    st.header("Métricas del modelo")

    metricas = cargar_metricas()
    threshold_data = cargar_threshold()

    if metricas is None:
        st.warning(
            "No se encontraron las métricas. Ejecuta primero "
            "model_training_evaluation.py."
        )
    else:
        # Modelo ganador y configuración
        st.subheader("Modelo seleccionado")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Modelo ganador", metricas["modelo_ganador"])
        with col_b:
            th = threshold_data["threshold_optimo"] if threshold_data else "N/A"
            st.metric("Umbral óptimo", f"{th}")
        with col_c:
            hiperparams = metricas.get("hiperparametros", {})
            st.markdown("**Hiperparámetros:**")
            for k, v in hiperparams.items():
                valor = f"{v:.4f}" if isinstance(v, float) else str(v)
                st.text(f"{k} = {valor}")

        # Métricas principales del modelo ganador
        st.subheader("Desempeño en el conjunto de prueba")
        m = metricas["metricas_test_threshold_default"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ROC-AUC", f"{m['roc_auc']:.4f}")
        c2.metric("F1-Score", f"{m['f1']:.4f}")
        c3.metric("PR-AUC", f"{m['pr_auc']:.4f}")
        c4.metric("Recall clase 0", f"{m['recall_clase_0']:.4f}")

        st.caption(
            "El recall de clase 0 mide la capacidad de detectar a los clientes "
            "que NO pagan (el objetivo de negocio más relevante)."
        )

        # Tabla comparativa de modelos
        st.subheader("Comparación de modelos")
        tabla_comp = pd.DataFrame(metricas["tabla_comparativa"])
        st.dataframe(tabla_comp, use_container_width=True, hide_index=True)

        # Galería de imágenes del entrenamiento
        st.subheader("Visualizaciones del entrenamiento")
        imagenes = [
            ("roc_curve_comparativa.png", "Curvas ROC"),
            ("precision_recall_curve_comparativa.png", "Curvas Precision-Recall"),
            ("matriz_confusion_ganador_threshold_default.png", "Matriz de confusión (umbral 0.5)"),
            ("matriz_confusion_ganador_threshold_optimizado.png", "Matriz de confusión (umbral óptimo)"),
            ("feature_importance.png", "Importancia de features"),
            ("curva_costo_vs_threshold.png", "Costo vs umbral"),
        ]

        # Mostrar en dos columnas
        for i in range(0, len(imagenes), 2):
            cols = st.columns(2)
            for col, (archivo, titulo) in zip(cols, imagenes[i:i+2]):
                ruta_img = RUTA_REPORTS_MODEL / archivo
                if ruta_img.exists():
                    with col:
                        st.markdown(f"**{titulo}**")
                        st.image(str(ruta_img), use_container_width=True)
