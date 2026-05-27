# artifacts/models/

Contiene el modelo ganador entrenado y los metadatos asociados:

- `modelo_ganador.pkl` — modelo serializado con `joblib`.
- `threshold_optimo.json` — umbral de decisión optimizado por análisis de costo.
- `metricas_finales.json` — métricas de desempeño del modelo en el set de prueba.

Estos archivos se generan al ejecutar:

```bash
python mlops_pipeline/src/model_training_evaluation.py
```
