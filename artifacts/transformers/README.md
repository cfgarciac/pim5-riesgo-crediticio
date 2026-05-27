# artifacts/transformers/

Contiene el pipeline de ingeniería de características entrenado, serializado con `joblib`:

- `pipeline_ft_engineering.pkl` — pipeline sklearn completo (imputers, encoders, scalers).
- `feature_names.json` — lista ordenada de nombres de features generadas (necesaria para el despliegue).

Estos artefactos se cargarán en producción para aplicar exactamente las mismas transformaciones a los datos nuevos antes de la inferencia.
