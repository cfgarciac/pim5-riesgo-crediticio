"""
test_ft_engineering.py
Pruebas unitarias para las funciones y clases de ``ft_engineering.py``.

Cubre:
    - Funciones de limpieza (tendencia_ingresos, capping, categóricas
      codificadas).
    - Transformers personalizados (NullIndicatorTransformer, Log1pTransformer).
    - Construcción del pipeline completo.
"""

import numpy as np
import pandas as pd
import pytest

from ft_engineering import (
    NullIndicatorTransformer,
    Log1pTransformer,
    limpiar_tendencia_ingresos,
    aplicar_capping_anomalias,
    convertir_categoricas_codificadas_a_string,
    construir_pipeline_ft_engineering,
)


# =========================================================================
# FIXTURES (datos reutilizables entre pruebas)
# =========================================================================

@pytest.fixture
def df_minimo():
    """Crea un DataFrame mínimo con todas las columnas que el pipeline
    espera, para usarlo en pruebas de integración."""
    return pd.DataFrame({
        "tipo_credito": [1, 2, 3],
        "capital_prestado": [1_000_000.0, 2_000_000.0, 3_000_000.0],
        "plazo_meses": [12, 24, 36],
        "edad_cliente": [30, 45, 60],
        "tipo_laboral": ["Empleado", "Independiente", "Empleado"],
        "salario_cliente": [2_000_000.0, 5_000_000.0, 8_000_000.0],
        "total_otros_prestamos": [0.0, 1_000_000.0, 2_000_000.0],
        "cuota_pactada": [100_000.0, 200_000.0, 300_000.0],
        "puntaje_datacredito": [500.0, 700.0, 800.0],
        "cant_creditosvigentes": [1, 3, 5],
        "huella_consulta": [0, 2, 4],
        "saldo_mora": [0.0, 100_000.0, 0.0],
        "saldo_total": [500_000.0, 1_000_000.0, 2_000_000.0],
        "saldo_principal": [500_000.0, 1_000_000.0, 2_000_000.0],
        "saldo_mora_codeudor": [0.0, 0.0, 1.0],
        "creditos_sectorFinanciero": [1, 2, 3],
        "creditos_sectorCooperativo": [0, 1, 0],
        "creditos_sectorReal": [0, 0, 1],
        "promedio_ingresos_datacredito": [1_500_000.0, 4_000_000.0, 7_000_000.0],
        "tendencia_ingresos": ["Creciente", "Estable", "Decreciente"],
    })


# =========================================================================
# limpiar_tendencia_ingresos
# =========================================================================

class TestLimpiarTendenciaIngresos:
    """Pruebas para la función limpiar_tendencia_ingresos."""

    def test_nan_se_convierte_en_desconocido(self):
        df = pd.DataFrame({"tendencia_ingresos": [np.nan, "Creciente"]})
        resultado = limpiar_tendencia_ingresos(df)
        assert resultado["tendencia_ingresos"].iloc[0] == "DESCONOCIDO"

    def test_categorias_validas_se_conservan(self):
        df = pd.DataFrame({
            "tendencia_ingresos": ["Creciente", "Estable", "Decreciente"]
        })
        resultado = limpiar_tendencia_ingresos(df)
        assert resultado["tendencia_ingresos"].tolist() == [
            "Creciente", "Estable", "Decreciente"
        ]

    def test_ruido_numerico_se_convierte_en_otros(self):
        df = pd.DataFrame({"tendencia_ingresos": [123, -456, "Creciente"]})
        resultado = limpiar_tendencia_ingresos(df)
        assert resultado["tendencia_ingresos"].iloc[0] == "OTROS"
        assert resultado["tendencia_ingresos"].iloc[1] == "OTROS"
        assert resultado["tendencia_ingresos"].iloc[2] == "Creciente"

    def test_no_modifica_el_dataframe_original(self):
        df = pd.DataFrame({"tendencia_ingresos": [np.nan]})
        df_original = df.copy()
        limpiar_tendencia_ingresos(df)
        # El df original no debe cambiar
        assert df["tendencia_ingresos"].isna().iloc[0]
        pd.testing.assert_frame_equal(df, df_original)


# =========================================================================
# aplicar_capping_anomalias
# =========================================================================

class TestAplicarCappingAnomalias:
    """Pruebas para la función aplicar_capping_anomalias."""

    def test_edad_fuera_de_rango_se_convierte_a_nan(self):
        df = pd.DataFrame({
            "edad_cliente": [15, 30, 120],  # 15 < 18, 120 > 100
        })
        resultado = aplicar_capping_anomalias(df)
        assert pd.isna(resultado["edad_cliente"].iloc[0])  # 15 -> NaN
        assert resultado["edad_cliente"].iloc[1] == 30  # válida
        assert pd.isna(resultado["edad_cliente"].iloc[2])  # 120 -> NaN

    def test_salario_excesivo_se_capea(self):
        df = pd.DataFrame({
            "salario_cliente": [-100, 5_000_000, 200_000_000],
        })
        resultado = aplicar_capping_anomalias(df)
        # Negativo -> 0 (cap), excesivo -> 100M (cap)
        assert resultado["salario_cliente"].iloc[0] == 0
        assert resultado["salario_cliente"].iloc[1] == 5_000_000
        assert resultado["salario_cliente"].iloc[2] == 100_000_000

    def test_puntaje_datacredito_cero_se_convierte_a_nan(self):
        df = pd.DataFrame({
            "puntaje_datacredito": [0, 500, 1500],
        })
        resultado = aplicar_capping_anomalias(df)
        # 0 < 1 -> NaN (codigo de "no reportado"), 500 ok, 1500 > 950 -> 950
        assert pd.isna(resultado["puntaje_datacredito"].iloc[0])
        assert resultado["puntaje_datacredito"].iloc[1] == 500
        assert resultado["puntaje_datacredito"].iloc[2] == 950

    def test_columnas_ausentes_no_rompen(self):
        # Si una variable de REGLAS_ANOMALIAS no está, la función no falla
        df = pd.DataFrame({"edad_cliente": [30, 40]})
        resultado = aplicar_capping_anomalias(df)
        assert len(resultado) == 2


# =========================================================================
# convertir_categoricas_codificadas_a_string
# =========================================================================

class TestConvertirCategoricas:
    """Pruebas para convertir_categoricas_codificadas_a_string."""

    def test_enteros_se_convierten_a_string(self):
        df = pd.DataFrame({
            "tipo_credito": [1, 2, 3],
            "saldo_mora_codeudor": [0, 1, 2],
        })
        resultado = convertir_categoricas_codificadas_a_string(df)
        assert resultado["tipo_credito"].iloc[0] == "1"
        assert isinstance(resultado["tipo_credito"].iloc[0], str)

    def test_nan_se_preserva(self):
        df = pd.DataFrame({
            "tipo_credito": [1.0, np.nan, 3.0],
            "saldo_mora_codeudor": [np.nan, 1.0, np.nan],
        })
        resultado = convertir_categoricas_codificadas_a_string(df)
        assert pd.isna(resultado["tipo_credito"].iloc[1])
        assert pd.isna(resultado["saldo_mora_codeudor"].iloc[0])


# =========================================================================
# NullIndicatorTransformer
# =========================================================================

class TestNullIndicatorTransformer:
    """Pruebas para el Transformer NullIndicatorTransformer."""

    def test_aprende_columnas_con_nulos_sobre_umbral(self):
        df = pd.DataFrame({
            "col_pocos_nulos": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # 0% nulos
            "col_muchos_nulos": [np.nan] * 5 + [1, 2, 3, 4, 5],  # 50% nulos
        })
        t = NullIndicatorTransformer(umbral_pct=10.0)
        t.fit(df)
        assert "col_muchos_nulos" in t.columnas_con_indicador_
        assert "col_pocos_nulos" not in t.columnas_con_indicador_

    def test_transform_agrega_columna_indicadora(self):
        df = pd.DataFrame({
            "col_muchos_nulos": [np.nan, np.nan, 1.0, 2.0, np.nan],
        })
        t = NullIndicatorTransformer(umbral_pct=5.0)
        t.fit(df)
        resultado = t.transform(df)
        # Debe haberse agregado la columna indicadora
        assert "col_muchos_nulos_fue_imputado" in resultado.columns
        # El indicador debe ser 1 donde había NaN, 0 donde no
        assert resultado["col_muchos_nulos_fue_imputado"].tolist() == [1, 1, 0, 0, 1]

    def test_get_feature_names_out(self):
        df = pd.DataFrame({"a": [1, 2], "b": [np.nan, np.nan]})
        t = NullIndicatorTransformer(umbral_pct=10.0)
        t.fit(df)
        nombres = list(t.get_feature_names_out())
        assert "a" in nombres
        assert "b" in nombres
        assert "b_fue_imputado" in nombres


# =========================================================================
# Log1pTransformer
# =========================================================================

class TestLog1pTransformer:
    """Pruebas para el Transformer Log1pTransformer."""

    def test_aplica_log1p_a_columnas_indicadas(self):
        df = pd.DataFrame({"x": [0.0, 1.0, 9.0], "y": [10.0, 20.0, 30.0]})
        t = Log1pTransformer(columnas=["x"])
        t.fit(df)
        resultado = t.transform(df)
        # log1p(0)=0, log1p(1)=ln(2), log1p(9)=ln(10)
        assert resultado["x"].iloc[0] == pytest.approx(0.0)
        assert resultado["x"].iloc[1] == pytest.approx(np.log(2))
        assert resultado["x"].iloc[2] == pytest.approx(np.log(10))
        # 'y' no se transforma
        assert resultado["y"].tolist() == [10.0, 20.0, 30.0]

    def test_preserva_nan(self):
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
        t = Log1pTransformer(columnas=["x"])
        t.fit(df)
        resultado = t.transform(df)
        assert pd.isna(resultado["x"].iloc[1])

    def test_columna_inexistente_lanza_error(self):
        df = pd.DataFrame({"x": [1.0, 2.0]})
        t = Log1pTransformer(columnas=["columna_que_no_existe"])
        with pytest.raises(ValueError, match="no encontradas"):
            t.fit(df)


# =========================================================================
# construir_pipeline_ft_engineering (prueba de integración)
# =========================================================================

class TestPipelineCompleto:
    """Prueba de integración: el pipeline completo fit + transform."""

    def test_pipeline_devuelve_dataframe_con_features_esperadas(self, df_minimo):
        # Aplicar la limpieza requerida antes del pipeline
        df = limpiar_tendencia_ingresos(df_minimo)
        df = aplicar_capping_anomalias(df)
        df = convertir_categoricas_codificadas_a_string(df)

        pipeline = construir_pipeline_ft_engineering()
        resultado = pipeline.fit_transform(df)

        # El pipeline debe devolver un DataFrame
        assert isinstance(resultado, pd.DataFrame)
        # Y tener el mismo número de filas
        assert len(resultado) == len(df)
        # Y al menos las features básicas (continuas + categóricas codificadas)
        assert resultado.shape[1] > 10

    def test_pipeline_no_genera_nan(self, df_minimo):
        df = limpiar_tendencia_ingresos(df_minimo)
        df = aplicar_capping_anomalias(df)
        df = convertir_categoricas_codificadas_a_string(df)

        pipeline = construir_pipeline_ft_engineering()
        resultado = pipeline.fit_transform(df)

        # Después del pipeline no debe quedar ningún NaN
        assert resultado.isnull().sum().sum() == 0
