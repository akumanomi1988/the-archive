param(
    [string]$Version = "v0.1.0",
    [switch]$OneFile = $true
)

$ErrorActionPreference = 'Stop'

# Project information
$ProjectName = "The Archive"
$Author = "akumanomi1988"
$GitHubUrl = "https://github.com/akumanomi1988/the-archive"
$BuildDate = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

Write-Host "🚀 Starting build and deployment process: $ProjectName $Version" -ForegroundColor Green

# Directories
$DistDir = "dist"
$OutputDir = "$DistDir\the-archive"

# 1. 🧹 Clean old dist directory
Write-Host "🧹 Cleaning old dist directory..."
if (Test-Path $DistDir) {
    Remove-Item -Recurse -Force $DistDir
    Write-Host "✅ Directory $DistDir deleted"
}

# 2. 🔨 Create output directory
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
Write-Host "📁 Output directory created: $OutputDir"

# 3. 📦 Files to copy to output directory
$FilesToCopy = @(
    "config.json",
    "skald.db",
    "enrichment_model.json",
    "library",
    "covers",
    "frontend",
    "index.html",
    "biblioteca.svg",
    "favicon-256.png",
    "favicon.ico"
)

Write-Host "📋 Copying files to output directory..."
foreach ($file in $FilesToCopy) {
    if (Test-Path $file) {
        if (Test-Path $file -PathType Container) {
            Copy-Item $file $OutputDir -Recurse -Force
            Write-Host "📁 Copied directory: $file"
        } else {
            Copy-Item $file $OutputDir -Force
            Write-Host "📄 Copied file: $file"
        }
    } else {
        Write-Warning "⚠️ Not found: $file"
        if ($file -eq "config.json") {
            $defaultConfig = @{
                library_path = "./library"
                db_path = "./skald.db"
                lm_enabled = $false
                lm_url = "http://127.0.0.1:1234/v1/chat/completions"
                lm_timeout = 300.0
                lm_mock = $true
                cors_origins = @("*")
                page_size_default = 20
                enrichment_batch_size = 10
                default_language = "es"
                lm_model = "openai/gpt-oss-20"
                lm_max_concurrency = 1
                lm_min_interval_ms = 10000
                providers_order = @("openlibrary", "googlebooks", "g4f", "lmstudio")
                openlibrary_enabled = $true
                openlibrary_timeout = 8.0
                googlebooks_enabled = $true
                googlebooks_timeout = 8.0
                g4f_enabled = $true
                g4f_model = "gpt-4o-mini"
                g4f_timeout = 30.0
            } | ConvertTo-Json -Depth 5
            $configPath = Join-Path $OutputDir "config.json"
            Set-Content -Path $configPath -Value $defaultConfig -Encoding UTF8
            Write-Host "✅ config.json created with defaults"
        }
        elseif ($file -eq "skald.db") {
            $dbPath = Join-Path $OutputDir "skald.db"
            New-Item -Path $dbPath -ItemType File -Force | Out-Null
            Write-Host "✅ skald.db created empty"
        }
    }
}

# 4. 🔧 Ensure covers/ exists
$coversDir = Join-Path $OutputDir "covers"
if (!(Test-Path $coversDir)) {
    New-Item -ItemType Directory -Path $coversDir | Out-Null
    Write-Host "📁 Directory covers/ created"
}

# 5. 📄 Create build info file
$infoPath = Join-Path $OutputDir "BUILD_INFO.txt"
$infoContent = @"
Project: $ProjectName
Version: $Version
Author: $Author
GitHub: $GitHubUrl
Build Date: $BuildDate
Status: Ready for distribution

This is a standalone executable for Windows.
Place this folder in any location and run the-archive.exe.

Configuration:
- Edit config.json to set your library path
- The database (skald.db) will be created/updated here
- Covers will be stored in the covers/ folder
"@
Set-Content -Path $infoPath -Value $infoContent -Encoding UTF8
Write-Host "📄 Build information created: BUILD_INFO.txt"

# 6. 🔧 Install PyInstaller
Write-Host "🔧 Installing PyInstaller..."
& python -m pip install --upgrade pyinstaller==6.10.0 pyinstaller-hooks-contrib==2024.8

# 7. 🔨 Run PyInstaller
Write-Host "🔨 Running PyInstaller..."
$distName = 'the-archive'
$pyiArgs = @('--noconfirm', "--name=$distName")

if ($OneFile) {
    $pyiArgs += '--onefile'
} else {
    $pyiArgs += '--onedir'
}

# Hidden imports
$pyiArgs += '--hidden-import'; $pyiArgs += 'backend'
$pyiArgs += '--hidden-import'; $pyiArgs += 'fastapi'
$pyiArgs += '--hidden-import'; $pyiArgs += 'uvicorn'
$pyiArgs += '--hidden-import'; $pyiArgs += 'sqlmodel'
$pyiArgs += '--hidden-import'; $pyiArgs += 'ebooklib'
$pyiArgs += '--hidden-import'; $pyiArgs += 'bs4'
$pyiArgs += '--hidden-import'; $pyiArgs += 'bleach'
$pyiArgs += '--hidden-import'; $pyiArgs += 'pydantic'
$pyiArgs += '--hidden-import'; $pyiArgs += 'pydantic_core'
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
$pyiArgs += '--hidden-import'; $pyiArgs += 'charset_normalizer'
$pyiArgs += '--hidden-import'; $pyiArgs += 'charset_normalizer.md'
$pyiArgs += '--hidden-import'; $pyiArgs += 'charset_normalizer.models'
$pyiArgs += '--hidden-import'; $pyiArgs += 'charset_normalizer.cd'
$pyiArgs += '--hidden-import'; $pyiArgs += 'charset_normalizer.utils'
$pyiArgs += '--hidden-import'; $pyiArgs += 'charset_normalizer.constant'

# Additional data
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
foreach ($d in $datas) {
    $pyiArgs += "--add-data"
    $pyiArgs += $d
}

# Entry point
$pyiArgs += "run_app.py"

# Execute PyInstaller
try {
    Write-Host "🔧 Executing: pyinstaller $pyiArgs"
    & pyinstaller @pyiArgs
    Write-Host "✅ PyInstaller executed successfully" -ForegroundColor Green
}
catch {
    Write-Error "❌ Error running PyInstaller: $_"
    exit 1
}

# 8. ✅ Final verification
$exePath = if ($OneFile) { Join-Path $DistDir "the-archive.exe" } else { Join-Path $OutputDir "the-archive.exe" }
Write-Host "🔍 Checking for executable at: $exePath"
Start-Sleep -Milliseconds 1000  # Wait briefly to ensure file system sync
if (Test-Path $exePath) {
    $fileInfo = Get-Item $exePath
    $fileSize = [math]::Round($fileInfo.Length / 1KB, 1)
    
    Write-Host "🎉 Build completed successfully!" -ForegroundColor Green
    Write-Host "📦 Executable generated: $exePath"
    Write-Host "📏 Size: $fileSize KB"
    Write-Host "📅 Build Date: $BuildDate"
    Write-Host "🔖 Version: $Version"
    Write-Host "👤 Author: $Author"
    Write-Host "🔗 GitHub: $GitHubUrl"
    
    Write-Host "`n📁 Files included in the package:"
    Get-ChildItem $OutputDir | ForEach-Object {
        if ($_.PSIsContainer) {
            Write-Host "📁 $($_.Name)/"
        } else {
            $size = [math]::Round($_.Length / 1KB, 1)
            Write-Host "📄 $($_.Name) ($size KB)"
        }
    }
    
    Write-Host "`n✅ Package is ready for distribution in the 'dist/' folder"
    Write-Host "⚠️ Check build\the-archive\warn-the-archive.txt for any dependency warnings."
}
else {
    Write-Error "❌ Executable was not generated at $exePath. Check build\the-archive\warn-the-archive.txt for details."
    exit 1
}