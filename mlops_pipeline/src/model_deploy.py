"""
Módulo: model_deploy.py
Propósito: API REST para el despliegue del modelo de riesgo crediticio.

Expone el modelo entrenado mediante una API desarrollada con FastAPI,
permitiendo obtener predicciones de riesgo crediticio a partir de los datos
de un cliente enviados en formato JSON.

Endpoints
---------
- GET  /         : verificación básica de que la API está activa.
- GET  /health   : estado detallado (modelo y pipeline cargados).
- POST /predict  : recibe los datos de un cliente y devuelve la predicción.
- GET  /docs     : documentación interactiva (generada automáticamente).

Ejecución local
---------------
Desde la raíz del repositorio:

    uvicorn mlops_pipeline.src.model_deploy:app --reload

O bien:

    python mlops_pipeline/src/model_deploy.py

Autor: Cristian García
Fecha de creación: 2026-05-28
Versión: V1.6.0 (esqueleto inicial)
"""

# =========================================================================
# 1. IMPORTS Y CONFIGURACIÓN DE RUTAS
# =========================================================================
import json
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException

# Configuración de rutas. El archivo está en mlops_pipeline/src/, por lo que
# la raíz del proyecto está dos niveles arriba.
RUTA_RAIZ = Path(__file__).resolve().parents[2]
RUTA_SRC = RUTA_RAIZ / "mlops_pipeline" / "src"

# Para cargar el pipeline serializado (que contiene Transformers
# personalizados), Python debe poder importar sus definiciones.
if str(RUTA_SRC) not in sys.path:
    sys.path.insert(0, str(RUTA_SRC))

try:
    from ft_engineering import (
        NullIndicatorTransformer,
        Log1pTransformer,
        limpiar_tendencia_ingresos,
        aplicar_capping_anomalias,
        convertir_categoricas_codificadas_a_string,
    )
    FT_ENGINEERING_OK = True
except ImportError:
    NullIndicatorTransformer = None
    Log1pTransformer = None
    FT_ENGINEERING_OK = False

# El pipeline se serializó cuando ft_engineering.py corría como script
# principal, por lo que las clases de los Transformers quedaron bajo el
# módulo '__main__'. Registramos esas clases en __main__ para que joblib
# pueda reconstruir el pipeline. (Misma solución usada en el dashboard.)
if FT_ENGINEERING_OK:
    import __main__
    __main__.NullIndicatorTransformer = NullIndicatorTransformer
    __main__.Log1pTransformer = Log1pTransformer
    logging.getLogger("ft_engineering").setLevel(logging.WARNING)


# =========================================================================
# 2. CONSTANTES DE RUTAS A ARTEFACTOS
# =========================================================================
RUTA_ARTIFACTS = RUTA_RAIZ / "artifacts"
RUTA_MODELS = RUTA_ARTIFACTS / "models"
RUTA_TRANSFORMERS = RUTA_ARTIFACTS / "transformers"

COL_TARGET = "Pago_atiempo"


# =========================================================================
# 3. CONFIGURACIÓN DE LOGGING
# =========================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("model_deploy")


# =========================================================================
# 4. CARGA DE ARTEFACTOS (al iniciar la API)
# =========================================================================

def cargar_artefactos() -> dict:
    """
    Carga el modelo, el pipeline y el threshold óptimo.

    Returns
    -------
    dict
        Diccionario con 'modelo', 'pipeline' y 'threshold'. Si algún
        artefacto falta, su valor será None.
    """
    artefactos = {"modelo": None, "pipeline": None, "threshold": None}

    ruta_modelo = RUTA_MODELS / "modelo_ganador.pkl"
    if ruta_modelo.exists():
        artefactos["modelo"] = joblib.load(ruta_modelo)
        logger.info("Modelo cargado: %s", ruta_modelo.name)

    ruta_pipeline = RUTA_TRANSFORMERS / "pipeline_ft_engineering.pkl"
    if ruta_pipeline.exists():
        artefactos["pipeline"] = joblib.load(ruta_pipeline)
        logger.info("Pipeline cargado: %s", ruta_pipeline.name)

    ruta_threshold = RUTA_MODELS / "threshold_optimo.json"
    if ruta_threshold.exists():
        with open(ruta_threshold, "r", encoding="utf-8") as f:
            artefactos["threshold"] = json.load(f)["threshold_optimo"]
        logger.info("Threshold cargado: %.2f", artefactos["threshold"])

    return artefactos


# Cargar artefactos una sola vez al importar el módulo
ARTEFACTOS = cargar_artefactos()


# =========================================================================
# 5. ESQUEMA DE ENTRADA (Pydantic)
# =========================================================================
from pydantic import BaseModel, Field


class ClienteInput(BaseModel):
    """
    Esquema de los datos de entrada de un cliente para la predicción.

    Contiene las 20 variables crudas que el pipeline de feature engineering
    espera. FastAPI valida automáticamente los tipos y campos requeridos.
    """
    tipo_credito: int = Field(..., description="Tipo de crédito (codificado)")
    capital_prestado: float = Field(..., description="Capital prestado (COP)")
    plazo_meses: int = Field(..., description="Plazo del crédito en meses")
    edad_cliente: int = Field(..., description="Edad del cliente en años")
    tipo_laboral: str = Field(..., description="Tipo laboral (Empleado/Independiente)")
    salario_cliente: float = Field(..., description="Salario mensual (COP)")
    total_otros_prestamos: float = Field(..., description="Total de otros préstamos (COP)")
    cuota_pactada: float = Field(..., description="Cuota mensual pactada (COP)")
    puntaje_datacredito: float = Field(..., description="Puntaje Datacrédito")
    cant_creditosvigentes: int = Field(..., description="Cantidad de créditos vigentes")
    huella_consulta: int = Field(..., description="Número de consultas (huella)")
    saldo_mora: float = Field(..., description="Saldo en mora (COP)")
    saldo_total: float = Field(..., description="Saldo total (COP)")
    saldo_principal: float = Field(..., description="Saldo principal (COP)")
    saldo_mora_codeudor: float = Field(..., description="Saldo en mora del codeudor (codificado)")
    creditos_sectorFinanciero: int = Field(..., description="Créditos en sector financiero")
    creditos_sectorCooperativo: int = Field(..., description="Créditos en sector cooperativo")
    creditos_sectorReal: int = Field(..., description="Créditos en sector real")
    promedio_ingresos_datacredito: float = Field(..., description="Promedio de ingresos reportado")
    tendencia_ingresos: str = Field(..., description="Tendencia de ingresos")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "tipo_credito": 7,
                    "capital_prestado": 3692160.0,
                    "plazo_meses": 10,
                    "edad_cliente": 42,
                    "tipo_laboral": "Independiente",
                    "salario_cliente": 8000000.0,
                    "total_otros_prestamos": 2500000.0,
                    "cuota_pactada": 341296.0,
                    "puntaje_datacredito": 695.0,
                    "cant_creditosvigentes": 10,
                    "huella_consulta": 5,
                    "saldo_mora": 0.0,
                    "saldo_total": 51258.0,
                    "saldo_principal": 51258.0,
                    "saldo_mora_codeudor": 0.0,
                    "creditos_sectorFinanciero": 5,
                    "creditos_sectorCooperativo": 0,
                    "creditos_sectorReal": 0,
                    "promedio_ingresos_datacredito": 908526.0,
                    "tendencia_ingresos": "Estable"
                }
            ]
        }
    }


class PrediccionOutput(BaseModel):
    """Esquema de la respuesta de la predicción."""
    probabilidad_pago: float = Field(..., description="Probabilidad de pago a tiempo (0-1)")
    decision: str = Field(..., description="APROBAR o RECHAZAR")
    threshold_aplicado: float = Field(..., description="Umbral usado para la decisión")


# =========================================================================
# 6. CREACIÓN DE LA APP FASTAPI
# =========================================================================
app = FastAPI(
    title="API de Riesgo Crediticio",
    description="Predice la probabilidad de pago a tiempo de un cliente.",
    version="1.6.0",
)


# =========================================================================
# 7. ENDPOINTS
# =========================================================================

@app.get("/")
def raiz():
    """Verificación básica de que la API está activa."""
    return {
        "mensaje": "API de Riesgo Crediticio activa.",
        "documentacion": "/docs",
    }


@app.get("/health")
def health():
    """
    Estado detallado de la API: indica si los artefactos están cargados.
    """
    estado_ok = (
        ARTEFACTOS["modelo"] is not None
        and ARTEFACTOS["pipeline"] is not None
        and ARTEFACTOS["threshold"] is not None
    )
    return {
        "estado": "ok" if estado_ok else "incompleto",
        "modelo_cargado": ARTEFACTOS["modelo"] is not None,
        "pipeline_cargado": ARTEFACTOS["pipeline"] is not None,
        "threshold_cargado": ARTEFACTOS["threshold"] is not None,
    }


# Endpoint /predict se implementa en la Etapa 4.B


@app.post("/predict", response_model=PrediccionOutput)
def predict(cliente: ClienteInput):
    """
    Predice la probabilidad de pago a tiempo de un cliente.

    Recibe los datos del cliente en formato JSON, los procesa mediante el
    pipeline de feature engineering, aplica el modelo y devuelve la
    predicción junto con la decisión basada en el umbral óptimo.

    Parameters
    ----------
    cliente : ClienteInput
        Datos del cliente (validados por Pydantic).

    Returns
    -------
    PrediccionOutput
        Probabilidad de pago, decisión y umbral aplicado.
    """
    # Verificar que los artefactos están disponibles
    if (ARTEFACTOS["modelo"] is None or ARTEFACTOS["pipeline"] is None
            or ARTEFACTOS["threshold"] is None):
        raise HTTPException(
            status_code=503,
            detail="Artefactos del modelo no disponibles. "
                   "Ejecuta el pipeline de entrenamiento primero."
        )

    try:
        # Convertir el input a DataFrame de una fila
        datos = cliente.model_dump()
        df = pd.DataFrame([datos])

        # Aplicar la misma limpieza que en ft_engineering.py
        df = limpiar_tendencia_ingresos(df)
        df = aplicar_capping_anomalias(df)
        df = convertir_categoricas_codificadas_a_string(df)

        # Transformar con el pipeline y predecir
        X_transformado = ARTEFACTOS["pipeline"].transform(df)
        proba_paga = float(ARTEFACTOS["modelo"].predict_proba(X_transformado)[:, 1][0])

        # Aplicar el umbral óptimo
        threshold = ARTEFACTOS["threshold"]
        decision = "APROBAR" if proba_paga >= threshold else "RECHAZAR"

        logger.info("Predicción generada: proba=%.4f, decision=%s",
                    proba_paga, decision)

        return PrediccionOutput(
            probabilidad_pago=round(proba_paga, 4),
            decision=decision,
            threshold_aplicado=threshold,
        )

    except Exception as e:
        logger.error("Error en la predicción: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Error al generar la predicción: {str(e)}"
        )


# =========================================================================
# 8. PUNTO DE ENTRADA
# =========================================================================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
