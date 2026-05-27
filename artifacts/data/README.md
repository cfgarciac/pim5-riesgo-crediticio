# artifacts/data/

Contiene los datasets transformados por `ft_engineering.py`:

- `X_train.parquet` — features de entrenamiento (80%).
- `X_test.parquet` — features de prueba (20%).
- `y_train.parquet` — target de entrenamiento.
- `y_test.parquet` — target de prueba.

Estos archivos se generan automáticamente al ejecutar:

```bash
python mlops_pipeline/src/ft_engineering.py
```

**Nota:** se persisten en formato Parquet por su mejor performance y compresión frente a CSV.
