# Build script to create a Windows executable using PyInstaller
# Usage: Open PowerShell, activate the virtualenv, then run: .\scripts\build_windows.ps1

param(
    [int]$Port = 8000,
    [switch]$OneFile
)

$ErrorActionPreference = 'Stop'

Write-Host "Using Python: $(Get-Command python).Path"

# Ensure pyinstaller is installed in the venv
Write-Host "Installing PyInstaller into virtualenv..."
python -m pip install --upgrade pip
python -m pip install pyinstaller

# Build args
$entry = 'run_app.py'
$distName = 'the-archive'
$icon = '' # set path to .ico if you want, e.g. 'favicon.ico'

$pyiArgs = @('--noconfirm', "--name=$distName")
if ($OneFile) { $pyiArgs += '--onefile' } else { $pyiArgs += '--onedir' }
if ($icon) { $pyiArgs += "--icon=$icon" }
# Ensure these data files are included (templates, static files, DB, covers, assets)
# Use relative paths from project root
$datas = @(
    "index.html;.",
    "biblioteca.svg;.",
    "favicon-256.png;.",
    "favicon.ico;.",
    "config.json;.",
    "enrichment_model.json;.",
    "skald.db;.",
    "covers;covers",
    "library;library"
)
foreach ($d in $datas) { $pyiArgs += "--add-data"; $pyiArgs += $d }

# Entry point
$pyiArgs += $entry

Write-Host "Running PyInstaller with args: $pyiArgs"
pyinstaller @pyiArgs

Write-Host "Build finished. See dist\$distName (or dist\$distName.exe if --onefile)."
