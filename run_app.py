import os
import sys
from uvicorn import Config, Server
from pathlib import Path

def get_application_path():
    """Obtiene el directorio correcto donde se encuentra la aplicación."""
    if getattr(sys, 'frozen', False):
        # Ejecutable empaquetado: usa el directorio del .exe
        return Path(sys.executable).parent
    else:
        # Desarrollo: usa el directorio del script
        return Path(__file__).parent

# Establecer el directorio de trabajo al de la aplicación
application_path = get_application_path()
os.chdir(application_path)
sys.path.insert(0, str(application_path))

# Verificar que los archivos esenciales existan
required_files = ['config.json', 'skald.db']
for filename in required_files:
    file_path = application_path / filename
    if not file_path.exists():
        print(f"⚠️  Advertencia: {filename} no encontrado en {application_path}")
        # Crear archivo vacío si es config.json
        if filename == 'config.json':
            default_config = {
                "library_path": str(application_path / "library"),
                "db_path": str(application_path / "skald.db"),
                "lm_enabled": False,
                "lm_url": "http://127.0.0.1:1234/v1/chat/completions",
                "lm_timeout": 300.0,
                "lm_mock": True,
                "cors_origins": ["*"],
                "page_size_default": 20,
                "enrichment_batch_size": 10,
                "default_language": "es"
            }
            with open(file_path, 'w', encoding='utf-8') as f:
                import json
                json.dump(default_config, f, indent=2, ensure_ascii=False)
            print(f"✅ {filename} creado con configuración por defecto")

def main(host: str = "0.0.0.0", port: int = 8000):
    """Inicia el servidor FastAPI."""
    print(f"🚀 Iniciando The Archive desde: {application_path}")
    print(f"📁 Directorio de trabajo: {os.getcwd()}")
    
    # Asegurar que backend se pueda importar
    try:
        import backend
        print("✅ Módulo backend importado correctamente")
    except Exception as e:
        print(f"❌ Error al importar backend: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    config = Config(
        "backend:app",
        host=host,
        port=port,
        log_level="info",
        workers=1,
    )
    
    server = Server(config=config)
    server.run()

if __name__ == "__main__":
    main()