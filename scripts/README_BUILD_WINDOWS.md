# Building a bundled Windows executable

This project can be packaged into a Windows executable using PyInstaller.

Prerequisites
- Windows machine.
- Python 3.10+ installed (same major version as the project's virtualenv is recommended).
- A virtual environment containing the project's dependencies (see `requirements.txt`).

Quick steps (from project root in PowerShell)

```powershell
# 1. Activate venv
& .\.venv\Scripts\Activate.ps1

# 2. Install build-time tools
python -m pip install --upgrade pip
python -m pip install pyinstaller

# 3. Build (one-folder)
.\scripts\build_windows.ps1

# Or build as one-file (single exe)
.\scripts\build_windows.ps1 -OneFile
```

Output
- The built files will be under `dist\the-archive` (onedir) or `dist\the-archive.exe` (onefile).

Caveats
- Some dependencies (e.g. `lxml`, `Pillow`, `uvicorn[standard]`) may include native wheels; PyInstaller usually handles them but antivirus/SmartScreen may flag the final exe.
- The final binary may be large (tens to hundreds of MB).
- If you need to update embedded static files (covers, library, DB), rebuild or ship them alongside the exe in the `dist` folder.

If you want, I can attempt to run the build here, but packaging may take several minutes and may fail if the environment here lacks OS-level tool support. I recommend running the provided script locally.
