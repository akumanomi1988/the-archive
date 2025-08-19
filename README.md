# The Archive — Indexado, búsqueda y descarga de EPUB con enriquecimiento por LLM local (LMStudio)

Proyecto minimalista: un backend FastAPI (un solo archivo) + un `index.html` sencillo para buscar, ver y descargar EPUB. Los metadatos se guardan en SQLite con SQLModel. Enriquecimiento opcional vía un endpoint HTTP de LMStudio.

## Características

## Estructura mínima

## Requisitos

## Instalación
1. Crear y activar un entorno virtual (opcional pero recomendado).
2. Instalar dependencias.

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

## Configuración
Copiar `config.json.example` a `config.json` y ajustar rutas y opciones.


También puedes definir `SKALD_CONFIG` con la ruta de configuración:

```powershell
$env:SKALD_CONFIG = "c:\\ruta\\a\\config.json"
```

## Ejecutar
Inicia el backend. Puedes abrir `http://localhost:8000/` y el backend servirá `index.html` si está en la misma carpeta.

```powershell
# opción A: python directo
python backend.py

# opción B: uvicorn
uvicorn backend:app --host 0.0.0.0 --port 8000
```

Endpoints clave:

## Notas de seguridad

## EXAMPLES
Ejemplos de uso con PowerShell/curl:

```powershell
# Reindexar
curl -s -X POST http://localhost:8000/reindex -H "content-type: application/json" -d '{"mode":"sync"}'

# Buscar por título/autor
curl -s "http://localhost:8000/books?q=Foundation&page=1&page_size=5"

# Ver detalle
curl -s http://localhost:8000/books/1

# Descargar EPUB id=1
curl -OJ http://localhost:8000/download/1

# Abrir HTML ligero (se devuelve HTML)
curl -s http://localhost:8000/open/1

# Enriquecer (uno)
curl -s -X POST http://localhost:8000/enrich/1

# Enriquecer lote
curl -s -X POST http://localhost:8000/enrich/batch -H "content-type: application/json" -d '{"ids":[1,2,3]}'
```

## Desarrollo y tests

```powershell
pytest -q
```

## Logo y favicons

Coloca `biblioteca.svg` en la raíz del proyecto (ya está incluido). Hay un script helper `scripts/generate_icons.py` que intenta generar `favicon-256.png` y `favicon.ico` desde el SVG usando `cairosvg` + `Pillow`. En Windows es más sencillo exportar manualmente con Inkscape:

```powershell
# Con Inkscape instalado
& 'C:\Program Files\Inkscape\inkscape.exe' -o favicon-256.png -w 256 -h 256 biblioteca.svg
# Luego crea el .ico (por ejemplo con GIMP) o usa herramientas online.
```

Si prefieres usar el script Python, instala las dependencias nativas para Cairo (por ejemplo via MSYS2) y luego:

```powershell
python -m pip install cairosvg pillow
python .\scripts\generate_icons.py
```

## Licencia
MIT
