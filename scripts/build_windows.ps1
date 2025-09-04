param(
    [int]$Port = 8000,
    [switch]$OneFile
)

$ErrorActionPreference = 'Stop'

Write-Host "Using Python: $(Get-Command python).Path"

# Asegurar PyInstaller actualizado
Write-Host "Installing PyInstaller..."
python -m pip install --upgrade pyinstaller==6.10.0 pyinstaller-hooks-contrib==2024.8

# Argumentos
$entry = 'run_app.py'
$distName = 'the-archive'
$icon = ''

$pyiArgs = @('--noconfirm', "--name=$distName")
if ($OneFile) { $pyiArgs += '--onefile' } else { $pyiArgs += '--onedir' }
if ($icon) { $pyiArgs += "--icon=$icon" }

# Hidden imports específicos
$pyiArgs += '--hidden-import'; $pyiArgs += 'backend'
$pyiArgs += '--hidden-import'; $pyiArgs += 'fastapi'
$pyiArgs += '--hidden-import'; $pyiArgs += 'uvicorn'
$pyiArgs += '--hidden-import'; $pyiArgs += 'sqlmodel'
$pyiArgs += '--hidden-import'; $pyiArgs += 'ebooklib'
$pyiArgs += '--hidden-import'; $pyiArgs += 'bs4'
$pyiArgs += '--hidden-import'; $pyiArgs += 'bleach'
$pyiArgs += '--hidden-import'; $pyiArgs += 'pydantic'
$pyiArgs += '--hidden-import'; $pyiArgs += 'pydantic_core'

# Submódulos de httpx
$pyiArgs += '--hidden-import'; $pyiArgs += 'httpx._client'
$pyiArgs += '--hidden-import'; $pyiArgs += 'httpx._models'
$pyiArgs += '--hidden-import'; $pyiArgs += 'httpx._config'
$pyiArgs += '--hidden-import'; $pyiArgs += 'httpx._exceptions'
$pyiArgs += '--hidden-import'; $pyiArgs += 'httpx._status_codes'
$pyiArgs += '--hidden-import'; $pyiArgs += 'httpx._transports.asgi'
$pyiArgs += '--hidden-import'; $pyiArgs += 'httpx._transports.wsgi'
$pyiArgs += '--hidden-import'; $pyiArgs += 'httpx._types'
$pyiArgs += '--hidden-import'; $pyiArgs += 'httpx._urls'
$pyiArgs += '--hidden-import'; $pyiArgs += 'httpx._utils'

# charset_normalizer
$pyiArgs += '--hidden-import'; $pyiArgs += 'charset_normalizer'
$pyiArgs += '--hidden-import'; $pyiArgs += 'charset_normalizer.md'
$pyiArgs += '--hidden-import'; $pyiArgs += 'charset_normalizer.models'
$pyiArgs += '--hidden-import'; $pyiArgs += 'charset_normalizer.cd'
$pyiArgs += '--hidden-import'; $pyiArgs += 'charset_normalizer.utils'
$pyiArgs += '--hidden-import'; $pyiArgs += 'charset_normalizer.constant'

# Datos adicionales
$datas = @(
    "index.html;.",
    "biblioteca.svg;.",
    "favicon-256.png;.",
    "favicon.ico;.",
    "config.json;.",
    "enrichment_model.json;.",
    "skald.db;.",
    "covers;covers",
    "frontend;frontend",
    "library;library",
    "backend.py;."
)
foreach ($d in $datas) { $pyiArgs += "--add-data"; $pyiArgs += $d }

# Entry point
$pyiArgs += $entry

Write-Host "Running PyInstaller with args: $pyiArgs"
pyinstaller @pyiArgs

Write-Host "Build finished. See dist\$distName.exe"