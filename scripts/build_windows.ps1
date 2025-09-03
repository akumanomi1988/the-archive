param(
    [int]$Port = 8000,
    [switch]$OneFile
)

$ErrorActionPreference = 'Stop'

Write-Host "Using Python: $(Get-Command python).Path"

# Instalar PyInstaller
Write-Host "Installing PyInstaller..."
python -m pip install --upgrade pip
python -m pip install pyinstaller

# Argumentos
$entry = 'run_app.py'
$distName = 'the-archive'
$icon = ''  # e.g., 'favicon.ico'

$pyiArgs = @('--noconfirm', "--name=$distName")
if ($OneFile) { $pyiArgs += '--onefile' } else { $pyiArgs += '--onedir' }
if ($icon) { $pyiArgs += "--icon=$icon" }

# Módulos ocultos
$pyiArgs += '--hidden-import'; $pyiArgs += 'backend'
$pyiArgs += '--hidden-import'; $pyiArgs += 'fastapi'
$pyiArgs += '--hidden-import'; $pyiArgs += 'uvicorn'
$pyiArgs += '--hidden-import'; $pyiArgs += 'starlette'
$pyiArgs += '--hidden-import'; $pyiArgs += 'pydantic'
$pyiArgs += '--hidden-import'; $pyiArgs += 'sqlalchemy'
$pyiArgs += '--hidden-import'; $pyiArgs += 'anyio'
$pyiArgs += '--hidden-import'; $pyiArgs += 'websockets'

# Datos
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