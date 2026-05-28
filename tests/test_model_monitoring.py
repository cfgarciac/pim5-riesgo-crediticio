"""
test_model_monitoring.py
Pruebas unitarias para las funciones de ``model_monitoring.py``.

Cubre:
    - Funciones de cálculo de drift (PSI numérico y categórico, KS, Chi2).
    - Interpretación del PSI.
    - Inducción de drift artificial.
    - Limpieza mínima del dataset.
"""

import numpy as np
import pandas as pd
import pytest

from model_monitoring import (
    interpretar_psi,
    calcular_psi,
    calcular_psi_categorico,
    calcular_ks,
    calcular_chi2,
    inducir_drift,
    limpiar_minimo,
)


# =========================================================================
# interpretar_psi
# =========================================================================

class TestInterpretarPsi:
    """Pruebas para la función interpretar_psi."""

    def test_psi_bajo_es_sin_drift(self):
        assert interpretar_psi(0.05) == "sin_drift"
        assert interpretar_psi(0.0) == "sin_drift"
        assert interpretar_psi(0.099) == "sin_drift"

    def test_psi_intermedio_es_drift_moderado(self):
        assert interpretar_psi(0.10) == "drift_moderado"
        assert interpretar_psi(0.15) == "drift_moderado"
        assert interpretar_psi(0.249) == "drift_moderado"

    def test_psi_alto_es_drift_severo(self):
        assert interpretar_psi(0.25) == "drift_severo"
        assert interpretar_psi(0.50) == "drift_severo"
        assert interpretar_psi(5.0) == "drift_severo"


# =========================================================================
# calcular_psi (variables numéricas)
# =========================================================================

class TestCalcularPsi:
    """Pruebas para calcular_psi sobre variables numéricas."""

    def test_distribuciones_identicas_dan_psi_cero(self):
        # Misma distribución -> PSI debe estar cerca de 0
        rng = np.random.default_rng(42)
        ref = pd.Series(rng.normal(0, 1, 1000))
        igual = pd.Series(rng.normal(0, 1, 1000))
        resultado = calcular_psi(ref, igual)
        # No exactamente 0 por aleatoriedad, pero debe ser bajo
        assert resultado["psi"] < 0.10
        assert resultado["interpretacion"] == "sin_drift"

    def test_distribuciones_muy_distintas_dan_psi_alto(self):
        # Distribuciones con medias muy distintas -> PSI alto
        rng = np.random.default_rng(42)
        ref = pd.Series(rng.normal(0, 1, 1000))
        distinta = pd.Series(rng.normal(5, 2, 1000))
        resultado = calcular_psi(ref, distinta)
        assert resultado["psi"] > 0.25
        assert resultado["interpretacion"] == "drift_severo"

    def test_psi_devuelve_diccionario_con_claves_esperadas(self):
        ref = pd.Series([1, 2, 3, 4, 5] * 20)
        nue = pd.Series([1, 2, 3, 4, 5] * 20)
        resultado = calcular_psi(ref, nue)
        assert "psi" in resultado
        assert "interpretacion" in resultado
        assert isinstance(resultado["psi"], float)

    def test_psi_ignora_nulos(self):
        # Si hay NaN, no deben romper el cálculo
        ref = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0] * 20)
        nue = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0] * 20)
        resultado = calcular_psi(ref, nue)
        # No debe lanzar excepción y debe devolver un valor finito
        assert np.isfinite(resultado["psi"])

    def test_variable_constante_no_genera_drift(self):
        # Variable con un solo valor único: no hay drift medible
        ref = pd.Series([5.0] * 100)
        nue = pd.Series([5.0] * 100)
        resultado = calcular_psi(ref, nue)
        assert resultado["psi"] == 0.0


# =========================================================================
# calcular_psi_categorico
# =========================================================================

class TestCalcularPsiCategorico:
    """Pruebas para calcular_psi_categorico."""

    def test_proporciones_iguales_dan_psi_bajo(self):
        ref = pd.Series(["A"] * 50 + ["B"] * 50)
        nue = pd.Series(["A"] * 50 + ["B"] * 50)
        resultado = calcular_psi_categorico(ref, nue)
        assert resultado["psi"] < 0.01

    def test_inversion_de_proporciones_da_psi_alto(self):
        # 70/30 -> 30/70 es un cambio fuerte
        ref = pd.Series(["A"] * 70 + ["B"] * 30)
        nue = pd.Series(["A"] * 30 + ["B"] * 70)
        resultado = calcular_psi_categorico(ref, nue)
        assert resultado["psi"] > 0.25
        assert resultado["interpretacion"] == "drift_severo"

    def test_categoria_nueva_no_rompe(self):
        # Si aparece una categoría en 'nuevos' que no estaba en referencia
        ref = pd.Series(["A"] * 80 + ["B"] * 20)
        nue = pd.Series(["A"] * 50 + ["B"] * 30 + ["C"] * 20)
        resultado = calcular_psi_categorico(ref, nue)
        # Debe ejecutarse sin error y dar un PSI finito
        assert np.isfinite(resultado["psi"])


# =========================================================================
# calcular_ks
# =========================================================================

class TestCalcularKs:
    """Pruebas para la prueba de Kolmogorov-Smirnov."""

    def test_misma_distribucion_p_value_alto(self):
        rng = np.random.default_rng(42)
        ref = pd.Series(rng.normal(0, 1, 500))
        igual = pd.Series(rng.normal(0, 1, 500))
        resultado = calcular_ks(ref, igual)
        # Distribuciones iguales: p > 0.05, no se rechaza H0
        assert resultado["p_value"] > 0.05
        assert resultado["hay_drift"] is False

    def test_distribuciones_distintas_p_value_bajo(self):
        rng = np.random.default_rng(42)
        ref = pd.Series(rng.normal(0, 1, 500))
        distinta = pd.Series(rng.normal(5, 1, 500))
        resultado = calcular_ks(ref, distinta)
        # Distribuciones muy distintas: p ~= 0
        assert resultado["p_value"] < 0.05
        assert resultado["hay_drift"] is True

    def test_devuelve_claves_esperadas(self):
        ref = pd.Series([1.0, 2.0, 3.0] * 50)
        nue = pd.Series([1.0, 2.0, 3.0] * 50)
        resultado = calcular_ks(ref, nue)
        assert "ks_statistic" in resultado
        assert "p_value" in resultado
        assert "hay_drift" in resultado


# =========================================================================
# calcular_chi2
# =========================================================================

class TestCalcularChi2:
    """Pruebas para Chi-cuadrado."""

    def test_misma_distribucion_no_detecta_drift(self):
        ref = pd.Series(["A"] * 60 + ["B"] * 40)
        igual = pd.Series(["A"] * 60 + ["B"] * 40)
        resultado = calcular_chi2(ref, igual)
        assert resultado["hay_drift"] is False
        assert resultado["p_value"] > 0.05

    def test_proporciones_distintas_detectan_drift(self):
        ref = pd.Series(["A"] * 80 + ["B"] * 20)
        distinta = pd.Series(["A"] * 20 + ["B"] * 80)
        resultado = calcular_chi2(ref, distinta)
        assert resultado["hay_drift"] is True
        assert resultado["p_value"] < 0.05

    def test_categoria_unica_no_rompe(self):
        # Si solo hay una categoría, Chi2 no aplica; debe manejarse
        ref = pd.Series(["A"] * 100)
        nue = pd.Series(["A"] * 100)
        resultado = calcular_chi2(ref, nue)
        # No debe lanzar error y debe devolver un valor sensato
        assert resultado["hay_drift"] is False


# =========================================================================
# inducir_drift
# =========================================================================

class TestInducirDrift:
    """Pruebas para la función inducir_drift."""

    @pytest.fixture
    def df_base(self):
        return pd.DataFrame({
            "edad_cliente": [30, 40, 50, 60, 70],
            "salario_cliente": [1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000],
            "tipo_laboral": ["Empleado"] * 4 + ["Independiente"],
            "otra_columna": [1, 2, 3, 4, 5],
        })

    def test_edad_se_incrementa(self, df_base):
        resultado = inducir_drift(df_base)
        # La edad debe haberse desplazado hacia arriba
        assert (resultado["edad_cliente"] > df_base["edad_cliente"]).all()

    def test_salario_se_multiplica(self, df_base):
        resultado = inducir_drift(df_base)
        # El salario debe ser mayor en todos los registros
        assert (resultado["salario_cliente"] > df_base["salario_cliente"]).all()

    def test_tipo_laboral_cambia_proporcion(self, df_base):
        # Construimos un df con muchos 'Empleado' para que el cambio sea visible
        df_grande = pd.DataFrame({
            "edad_cliente": [40] * 100,
            "salario_cliente": [3_000_000] * 100,
            "tipo_laboral": ["Empleado"] * 100,
        })
        resultado = inducir_drift(df_grande)
        # Después del drift, debe haber al menos algunos 'Independiente'
        assert (resultado["tipo_laboral"] == "Independiente").sum() > 0

    def test_otras_columnas_no_se_tocan(self, df_base):
        resultado = inducir_drift(df_base)
        # 'otra_columna' no está en la lista de variables a inducir
        pd.testing.assert_series_equal(
            resultado["otra_columna"], df_base["otra_columna"]
        )

    def test_no_modifica_el_input(self, df_base):
        df_original = df_base.copy()
        inducir_drift(df_base)
        pd.testing.assert_frame_equal(df_base, df_original)


# =========================================================================
# limpiar_minimo
# =========================================================================

class TestLimpiarMinimo:
    """Pruebas para la limpieza mínima usada en el monitoreo."""

    def test_tendencia_nulos_se_convierten_a_desconocido(self):
        df = pd.DataFrame({"tendencia_ingresos": [np.nan, "Creciente"]})
        resultado = limpiar_minimo(df)
        assert resultado["tendencia_ingresos"].iloc[0] == "DESCONOCIDO"

    def test_aplica_capping_de_edad(self):
        df = pd.DataFrame({
            "edad_cliente": [10, 30, 120],
        })
        resultado = limpiar_minimo(df)
        # 10 y 120 fuera de [18, 100] -> NaN
        assert pd.isna(resultado["edad_cliente"].iloc[0])
        assert resultado["edad_cliente"].iloc[1] == 30
        assert pd.isna(resultado["edad_cliente"].iloc[2])

    def test_columnas_ausentes_no_rompen(self):
        # Si una columna no está, la función no debe fallar
        df = pd.DataFrame({"otra_columna": [1, 2, 3]})
        resultado = limpiar_minimo(df)
        assert len(resultado) == 3
