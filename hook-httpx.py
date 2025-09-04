# hook-httpx.py
hiddenimports = [
    'httpx._client',
    'httpx._models',
    'httpx._config',
    'httpx._exceptions',
    'httpx._status_codes',
    'httpx._transports.asgi',
    'httpx._transports.wsgi',
    'httpx._types',
    'httpx._urls',
    'httpx._utils',
]

# Evita que PyInstaller excluya módulos importantes
excludedimports = []