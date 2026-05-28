"""
conftest.py
Configuración global de pytest.

Agrega ``mlops_pipeline/src`` al ``sys.path`` para que las pruebas puedan
importar los módulos del proyecto (ft_engineering, model_monitoring, etc.)
sin necesidad de instalar el proyecto como paquete.
"""
import sys
from pathlib import Path

RUTA_RAIZ = Path(__file__).resolve().parent.parent
RUTA_SRC = RUTA_RAIZ / "mlops_pipeline" / "src"

if str(RUTA_SRC) not in sys.path:
    sys.path.insert(0, str(RUTA_SRC))
