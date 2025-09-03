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
# Añade estas exclusiones antes de los hidden-imports
$pyiArgs += '--exclude-module'; $pyiArgs += 'test'
$pyiArgs += '--exclude-module'; $pyiArgs += 'unittest'
$pyiArgs += '--exclude-module'; $pyiArgs += 'setuptools'
$pyiArgs += '--exclude-module'; $pyiArgs += 'distutils'

# Luego incluye explícitamente los módulos necesarios
$pyiArgs += '--hidden-import'; $pyiArgs += 'certifi'
$pyiArgs += '--hidden-import'; $pyiArgs += 'charset_normalizer'
$pyiArgs += '--hidden-import'; $pyiArgs += 'sqlmodel'
$pyiArgs += '--hidden-import'; $pyiArgs += 'fastapi'
$pyiArgs += '--hidden-import'; $pyiArgs += 'uvicorn'
$pyiArgs += '--hidden-import'; $pyiArgs += 'httpx'
$pyiArgs += '--hidden-import'; $pyiArgs += 'ebooklib'
$pyiArgs += '--hidden-import'; $pyiArgs += 'bs4'
$pyiArgs += '--hidden-import'; $pyiArgs += 'bleach'
$pyiArgs += '--additional-hooks-dir'; $pyiArgs += '.'

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