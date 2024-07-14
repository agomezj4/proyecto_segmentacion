"""proyecto_segmentacion.pipelines
"""
__version__ = "0.1"

import os
import sys

# Añadir el directorio raíz del proyecto al sys.path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

# Importar módulos necesarios
from .utils import Utils
