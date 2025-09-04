# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['run_app.py'],
    pathex=[],
    binaries=[],
    datas=[('index.html', '.'), ('biblioteca.svg', '.'), ('favicon-256.png', '.'), ('favicon.ico', '.'), ('config.json', '.'), ('enrichment_model.json', '.'), ('skald.db', '.'), ('covers', 'covers'), ('frontend', 'frontend'), ('library', 'library'), ('backend.py', '.')],
    hiddenimports=['backend', 'fastapi', 'uvicorn', 'sqlmodel', 'ebooklib', 'bs4', 'bleach', 'pydantic', 'pydantic_core', 'httpx._client', 'httpx._models', 'httpx._config', 'httpx._exceptions', 'httpx._status_codes', 'httpx._transports.asgi', 'httpx._transports.wsgi', 'httpx._types', 'httpx._urls', 'httpx._utils', 'charset_normalizer', 'charset_normalizer.md', 'charset_normalizer.models', 'charset_normalizer.cd', 'charset_normalizer.utils', 'charset_normalizer.constant'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='the-archive',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='the-archive',
)
