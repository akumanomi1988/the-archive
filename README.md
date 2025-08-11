# Skald — Indexado, búsqueda y descarga de EPUB con enriquecimiento por LLM local (LMStudio)

Proyecto minimalista: un backend FastAPI (un solo archivo) + un `index.html` sencillo para buscar, ver y descargar EPUB. Los metadatos se guardan en SQLite con SQLModel. Enriquecimiento opcional vía un endpoint HTTP de LMStudio.

## Características
- Indexa una carpeta de biblioteca: `biblioteca/Autor_con_guiones_bajos/Titulo_con_guiones_bajos.epub`.
- Extrae: título, autor, idioma (si está), ruta, tamaño, fecha modificación ISO, sha256.
- Idempotencia por sha256.
- API REST para búsqueda/filtrado, detalle, visualización HTML ligera y descarga del EPUB.
- Enriquecimiento opcional (mock o real) con LMStudio vía HTTP POST.
- Frontend `index.html` con JS/CSS embebidos y Fetch API.

## Estructura mínima
- `backend.py` — API y lógica (config, modelos, DB, indexador, LLM client, rutas).
- `index.html` — UI mínima de búsqueda y visor.
- `config.json.example` — ejemplo de configuración.
- `requirements.txt`
- `tests.py` — pruebas básicas con pytest y httpx.

## Requisitos
- Python 3.11+

## Instalación
1. Crear y activar un entorno virtual (opcional pero recomendado).
2. Instalar dependencias.

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

## Configuración
Copiar `config.json.example` a `config.json` y ajustar rutas y opciones.

- `library_path`: carpeta donde están los `.epub`.
- `db_path`: ruta del archivo SQLite.
- `lm_enabled`: `true` para activar el enriquecimiento.
- `lm_url`: URL de LMStudio (por ejemplo `http://localhost:1234/api/predict`).
- `lm_timeout`: segundos de timeout.
- `lm_mock`: si `true`, responde con datos simulados.
- `cors_origins`: orígenes permitidos para CORS.
- `page_size_default`: tamaño de página por defecto.
- `enrichment_batch_size`: tamaño del lote de enriquecimiento.

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
- `GET http://localhost:8000/health`
- `POST http://localhost:8000/reindex` (body `{ "mode": "sync" }`)
- `GET http://localhost:8000/books?q=Asimov&autor=Isaac&page=1&page_size=20`
- `GET http://localhost:8000/books/{id}`
- `GET http://localhost:8000/open/{id}`
- `GET http://localhost:8000/download/{id}`
- `POST http://localhost:8000/enrich/{id}`
- `POST http://localhost:8000/enrich/batch` (body `{ "ids": [1,2,3] }` opcional)

## Notas de seguridad
- El contenido HTML renderizado desde EPUB se sanea con `bleach`, pero es recomendable usar bibliotecas confiables y no exponer el servicio a Internet sin endurecerlo.
- No se almacena el texto del libro en la base de datos.

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

## Licencia
MIT
