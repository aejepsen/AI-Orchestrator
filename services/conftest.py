"""Garante que os pacotes dos serviços (financas, rh, ...) sejam importáveis nos testes."""

import sys
from pathlib import Path

SERVICES_DIR = Path(__file__).resolve().parent
if str(SERVICES_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICES_DIR))
