# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['run_app.py'],
    pathex=[],
    binaries=[],
    datas=[('index.html', '.'), ('biblioteca.svg', '.'), ('favicon-256.png', '.'), ('favicon.ico', '.'), ('config.json', '.'), ('enrichment_model.json', '.'), ('skald.db', '.'), ('covers', 'covers'), ('frontend', 'frontend'), ('library', 'library'), ('backend.py', '.')],
    hiddenimports=['certifi', 'charset_normalizer', 'sqlmodel', 'fastapi', 'uvicorn', 'httpx', 'ebooklib', 'bs4', 'bleach'],
    hookspath=['.'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['test', 'unittest', 'setuptools', 'distutils'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='the-archive',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
