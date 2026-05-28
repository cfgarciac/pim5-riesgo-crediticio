# Hallazgos del Análisis Exploratorio de Datos (EDA)

**Proyecto:** PIM5 - Modelo de Riesgo Crediticio
**Dataset:** `Base_de_datos.xlsx` (10,763 registros, 23 columnas)
**Versión del documento:** V1.0.1
**Autor:** Cristian García

---

## 1. Resumen ejecutivo

El dataset histórico de créditos contiene 10,763 registros y 23 variables, sin duplicados.
El objetivo es predecir la variable binaria `Pago_atiempo` (1 = paga a tiempo, 0 = no paga).

Los puntos críticos identificados son:

1. **Desbalance severo del target:** 95.25% de clientes paga a tiempo (10,252) frente a
   4.75% que no paga (511). Este desbalance domina el diseño del modelado.
2. **Variables con altas proporciones de nulos:** `tendencia_ingresos` y
   `promedio_ingresos_datacredito` presentan ~27% de valores faltantes (probablemente
   provienen de la misma fuente externa).
3. **Asimetría extrema en variables financieras:** `salario_cliente` (skew=43.78),
   `saldo_mora` (skew=40.57), `total_otros_prestamos` (skew=38.46), entre otras, requieren
   transformación logarítmica antes del modelado.
4. **Valores anómalos por error de captura:** edad de 123 años, salario de 22 mil millones,
   puntaje DataCrédito negativo. Requieren capping al P99.
5. **Multicolinealidad:** algunos grupos de variables (saldos, capital-cuota) están
   altamente correlacionados entre sí.
6. **Patrón de impago no linealmente separable:** los pair plots y scatter plots muestran
   que las clases no se separan limpiamente en el espacio de las top features. Modelos
   no lineales (XGBoost, Random Forest) serán los candidatos principales.

---

## 2. Composición del dataset

### 2.1 Clasificación de variables

| Tipo | Cantidad | Variables |
|---|---|---|
| Target | 1 | `Pago_atiempo` |
| Fecha | 1 | `fecha_prestamo` |
| Categórica de texto | 2 | `tipo_laboral`, `tendencia_ingresos` |
| Categórica codificada | 2 | `tipo_credito`, `saldo_mora_codeudor` |
| Numérica discreta | 2 | `plazo_meses`, `creditos_sectorCooperativo` |
| Numérica continua | 15 | Resto de variables financieras y de comportamiento |

### 2.2 Valores nulos

| Variable | % Nulos |
|---|---|
| `tendencia_ingresos` | 27.24% |
| `promedio_ingresos_datacredito` | 27.22% |
| `saldo_mora_codeudor` | 5.48% |
| `saldo_principal` | 3.76% |
| `saldo_mora` | 1.45% |
| `saldo_total` | 1.45% |
| `puntaje_datacredito` | 0.06% |

Las dos primeras variables tienen conteos casi idénticos de nulos, sugiriendo que faltan
conjuntamente. Probablemente provienen de DataCrédito y son ausentes cuando ese servicio
no devuelve información del cliente.

### 2.3 Calidad de datos

- **Duplicados:** 0.
- **Anomalías por semántica de negocio:**
  - `edad_cliente` máximo de 123 años (biológicamente improbable).
  - `salario_cliente` máximo de 22,000,000,000 (error de captura claro).
  - `puntaje_datacredito` con valores negativos (la escala oficial va de 150 a 950).
  - `total_otros_prestamos` máximo de 6,787,675,263 (revisar si es centavos o pesos).
- **Variable mixta:** `tendencia_ingresos` mezcla categorías de texto (Creciente, Estable,
  Decreciente) con valores numéricos sueltos (~1% del total). Requiere limpieza.

---

## 3. Hallazgos por análisis

### 3.1 Análisis univariable

- **Distribuciones asimétricas positivas extremas** en la mayoría de variables monetarias.
  La transformación logarítmica (`log1p`) será esencial.
- **Distribuciones asimétricas negativas** en `puntaje` (skew=-4.87) y `puntaje_datacredito`
  (skew=-5.64): los valores se concentran en la zona alta del rango.
- **Outliers IQR abundantes** en variables muy sesgadas, pero la mayoría son valores
  legítimos en colas largas, no errores. NO eliminarlos masivamente.
- **Distribución temporal:** los préstamos se distribuyen de forma desigual mes a mes;
  el volumen varía pero no hay una tendencia monotónica clara.

### 3.2 Análisis bivariable

- **Tests estadísticos:** la mayoría de variables muestran significancia estadística (p-valor
  bajo) en Mann-Whitney U y Chi-cuadrado, **incluso después de aplicar corrección de
  Bonferroni**. Esto se debe en parte al tamaño de muestra grande.
- **Correlaciones point-biserial individuales son moderadas o débiles** (típicamente
  |r| < 0.3). El poder predictivo del modelo dependerá de combinar muchas señales débiles.
- **Categóricas informativas:**
  - `tipo_credito`: ciertos códigos muestran tasas de pago muy distintas a la global.
  - `tendencia_ingresos`: clientes con tendencia Decreciente presentan mayor tasa de
    impago, como cabría esperar.
  - `tipo_laboral`: la diferencia entre Empleado e Independiente es moderada pero presente.
- **Evolución temporal de la tasa de pago:** oscila alrededor del 95% global, sin tendencia
  estructural clara. Justifica monitoreo de drift en producción.

### 3.3 Análisis multivariable

- **Matriz de correlación Spearman:** correlaciones mayoritariamente moderadas o bajas
  entre features (lo cual es bueno: las variables aportan información parcialmente
  independiente).
- **Pares altamente correlacionados (|r| >= 0.7):**
  - Grupo de saldos: `saldo_total`, `saldo_principal`, `saldo_mora` (lógicamente
    relacionados por construcción).
  - `capital_prestado` y `cuota_pactada` (la cuota se deriva del capital).
- **VIF severo (>=10)** en variables involucradas en los pares altamente correlacionados.
  Manejable con regularización (Lasso/Ridge) o selección automática en árboles.
- **Pair plot:** las clases del target NO se separan linealmente en el espacio de las top
  features. La clase 0 (impago) aparece dispersa, no concentrada — patrón complejo.
- **Estabilidad temporal de features:** medianas mensuales relativamente estables sin
  tendencias claras de mediano plazo.

---

## 4. Tabla maestra de decisiones para Feature Engineering (V1.1.0)

| variable                      | tipo                  |   pct_nulos | accion_nulos                                | transformacion         | encoding                           | decision_final                                         |
|:------------------------------|:----------------------|------------:|:--------------------------------------------|:-----------------------|:-----------------------------------|:-------------------------------------------------------|
| tipo_credito                  | categorica_codificada |        0    | sin nulos                                   | -                      | One-hot drop_first=True (5 col)    | Conservar como categorica.                             |
| fecha_prestamo                | fecha                 |        0    | -                                           | -                      | -                                  | Excluir del modelo; conservar para monitoreo de drift. |
| capital_prestado              | numerica_continua     |        0    | sin nulos                                   | log1p (skew=3.7)       | -                                  | Aplicar transformacion + StandardScaler.               |
| plazo_meses                   | numerica_discreta     |        0    | sin nulos                                   | log1p (skew=2.5)       | -                                  | Aplicar transformacion + StandardScaler.               |
| edad_cliente                  | numerica_continua     |        0    | sin nulos                                   | ninguna                | -                                  | Capping P99 antes de log1p y escalado.                 |
| tipo_laboral                  | categorica_texto      |        0    | sin nulos                                   | -                      | Label encoding (1 columna binaria) | Conservar; encoding simple binario.                    |
| salario_cliente               | numerica_continua     |        0    | sin nulos                                   | log1p (skew=43.8)      | -                                  | Capping P99 antes de log1p y escalado.                 |
| total_otros_prestamos         | numerica_continua     |        0    | sin nulos                                   | log1p (skew=38.5)      | -                                  | Capping P99 antes de log1p y escalado.                 |
| cuota_pactada                 | numerica_continua     |        0    | sin nulos                                   | log1p (skew=3.8)       | -                                  | Aplicar transformacion + StandardScaler.               |
| puntaje                       | numerica_continua     |        0    | sin nulos                                   | sin transf (skew=-4.9) | -                                  | Capping P99 antes de log1p y escalado.                 |
| puntaje_datacredito           | numerica_continua     |        0.06 | imputar mediana                             | sin transf (skew=-5.6) | -                                  | Capping P99 antes de log1p y escalado.                 |
| cant_creditosvigentes         | numerica_continua     |        0    | sin nulos                                   | ninguna                | -                                  | Aplicar transformacion + StandardScaler.               |
| huella_consulta               | numerica_continua     |        0    | sin nulos                                   | ninguna                | -                                  | Aplicar transformacion + StandardScaler.               |
| saldo_mora                    | numerica_continua     |        1.45 | imputar mediana                             | log1p (skew=40.6)      | -                                  | Aplicar transformacion + StandardScaler.               |
| saldo_total                   | numerica_continua     |        1.45 | imputar mediana                             | log1p (skew=20.2)      | -                                  | Aplicar transformacion + StandardScaler.               |
| saldo_principal               | numerica_continua     |        3.76 | imputar mediana                             | log1p (skew=5.1)       | -                                  | Aplicar transformacion + StandardScaler.               |
| saldo_mora_codeudor           | categorica_codificada |        5.48 | NaN -> SIN_CODEUDOR                         | -                      | One-hot drop_first=True (3 col)    | Tratar como categorica.                                |
| creditos_sectorFinanciero     | numerica_continua     |        0    | sin nulos                                   | log1p (skew=2.7)       | -                                  | Aplicar transformacion + StandardScaler.               |
| creditos_sectorCooperativo    | numerica_discreta     |        0    | sin nulos                                   | log1p (skew=4.2)       | -                                  | Aplicar transformacion + StandardScaler.               |
| creditos_sectorReal           | numerica_continua     |        0    | sin nulos                                   | log1p (skew=3.2)       | -                                  | Aplicar transformacion + StandardScaler.               |
| promedio_ingresos_datacredito | numerica_continua     |       27.22 | imputar mediana + indicador                 | log1p (skew=4.3)       | -                                  | Aplicar transformacion + StandardScaler.               |
| tendencia_ingresos            | categorica_texto      |       27.24 | NaN -> DESCONOCIDO; ruido numerico -> OTROS | -                      | One-hot drop_first=True (3 col)    | Limpieza previa + indicador fue_imputado.              |
| Pago_atiempo                  | target                |        0    | -                                           | -                      | -                                  | No es predictor; es la variable a predecir.            |

---

## 5. Implicaciones para el modelado

### 5.1 Tipos de modelos a evaluar

- **Baseline lineal:** Regresión Logística con `class_weight='balanced'` y regularización L2.
  Sirve como punto de referencia.
- **Modelos basados en árboles:** Random Forest, XGBoost, LightGBM. Son los candidatos
  principales dado que:
  - El problema no es linealmente separable.
  - Manejan bien la multicolinealidad.
  - Son robustos a outliers y escalas distintas.
  - Permiten ajustar pesos por clase de forma efectiva.

### 5.2 Métricas de evaluación

| Métrica | Por qué se usa |
|---|---|
| ROC-AUC | Métrica principal exigida por el PI; robusta al desbalance |
| F1-score | Balance entre precision y recall |
| Precision-Recall AUC | Más informativa que ROC-AUC en datasets muy desbalanceados |
| Recall sobre clase 0 | Métrica de negocio: capacidad de detectar a los incumplidos |
| Matriz de confusión | Análisis cualitativo de errores (FN vs FP) |

**No se usará accuracy como métrica de decisión** (sería engañosa con 95% de clase 1).

### 5.3 Manejo del desbalance

- **Durante el entrenamiento:** `class_weight='balanced'` (regresión y RF), `scale_pos_weight=20.06` (XGBoost).
- **No se aplicará SMOTE** en V1.1.0. La clase minoritaria (511 muestras) es suficiente
  para que el peso de clase sea efectivo.
- **Threshold tuning:** se optimizará el umbral de decisión basado en función de costo de
  negocio, no se asumirá 0.5 por defecto.

---

## 6. Próximos pasos

1. **V1.1.0:** Implementar `ft_engineering.py` siguiendo la tabla maestra (Sección 4).
2. **V1.0.1 (modelado):** Entrenamiento y comparación de modelos en `model_training_evaluation.py`.
3. **Avance #3:** Monitoreo de data drift (`model_monitoring.py`) y dashboard en Streamlit.
4. **Avance #4:** Despliegue mediante FastAPI + Docker (`model_deploy.py`).

---

## 7. Hallazgo crítico: Target Leakage en la variable 'puntaje'

Durante la fase de modelado (Avance #2) se detectó un caso de **target leakage**
(fuga de información del objetivo) en la variable `puntaje`.

### 7.1 Síntoma

Al entrenar los primeros modelos, los cuatro algoritmos alcanzaron un
ROC-AUC de 1.0000 en validación cruzada. Un desempeño perfecto es una señal
de alerta clásica de que alguna variable contiene información del target que
no estaría disponible en un escenario real de predicción.

### 7.2 Diagnóstico

El análisis de correlación y separación de clases reveló:

- `puntaje` presentaba una correlación de 0.92 con `Pago_atiempo`.
- La variable separaba las clases de forma casi perfecta:
  - Registros con `puntaje > 75` correspondían siempre a `Pago_atiempo = 1`.
  - Registros con `puntaje <= 50` correspondían siempre a `Pago_atiempo = 0`.
  - Para la clase 1 (paga), el valor se concentraba alrededor de 95.
  - Para la clase 0 (no paga), el valor caía entre -17 y 62.

Esto indica que `puntaje` es un score calculado **a posteriori** del
comportamiento de pago (es decir, un resultado derivado de saber si el
cliente pagó), y no una variable disponible al momento de evaluar una
solicitud de crédito nueva.

### 7.3 Verificación de exhaustividad

Se revisaron todas las demás variables (numéricas y categóricas) para
descartar más casos de leakage:

- Ninguna otra variable superó una correlación de 0.13 con el target.
- `puntaje_datacredito` (correlación 0.12) es el score crediticio externo
  legítimo (tipo Datacredito), que SÍ se conoce antes de otorgar el crédito.
  Se conserva en el modelo.
- Las categorías con "100% de pago" en `tendencia_ingresos` correspondían al
  ruido numérico de baja frecuencia ya identificado en el EDA (1 a 7
  registros cada una), no a un patrón real. Se agrupan como OTROS.

Conclusión: el único caso de leakage era `puntaje`.

### 7.4 Corrección aplicada

En `ft_engineering.py` se agregó la constante `COLS_EXCLUIR_LEAKAGE = ["puntaje"]`
y la columna se incorporó a `COLS_DESCARTAR`, de modo que se elimina del
pipeline antes del entrenamiento. El dataset pasó de 33 a 32 features y la
correlación máxima con el target descendió a 0.124 (saldo_mora), un valor
sano. Tras la corrección, los modelos arrojaron ROC-AUC realistas en el
rango 0.65 a 0.72, coherentes con la dificultad intrínseca del problema.

### 7.5 Lección aprendida

Un desempeño perfecto o casi perfecto nunca debe celebrarse sin verificación.
En problemas de riesgo crediticio es habitual que existan variables derivadas
del resultado (scores internos, flags de comportamiento) que contaminan el
entrenamiento. La validación contra el sentido de negocio (qué información
existe realmente al momento de predecir) es tan importante como las métricas
estadísticas.

---

---
**Fecha de última actualización del EDA:** 2026-05-28
