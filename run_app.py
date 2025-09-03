# HACK: Solución para el bug de PyInstaller con Python 3.10
# Debe ir ANTES de cualquier otra importación
import sys
if getattr(sys, 'frozen', False):
    import os
    import importlib.util
    
    # Forzar la carga de ciertos módulos antes de que PyInstaller los analice
    def _force_import(name):
        spec = importlib.util.find_spec(name)
        if spec:
            importlib.util.module_from_spec(spec)
    
    _force_import('typing')
    _force_import('typing_extensions')
    _force_import('sqlmodel')
    _force_import('fastapi')
    _force_import('uvicorn')
    _force_import('httpx')
    _force_import('ebooklib')
    
import os
import sys
from uvicorn import Config, Server
from pathlib import Path

# Añade el directorio actual al PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importa explícitamente el módulo backend
import backend  # type: ignore

def main(host: str = "0.0.0.0", port: int = 8000):
    config = Config(
        "backend:app",
        host=host,
        port=port,
        log_level="info",
        workers=4,  # Usa 4 workers para mejorar el rendimiento
    )
    server = Server(config=config)
    server.run()

if __name__ == "__main__":
    main()