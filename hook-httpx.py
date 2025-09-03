# hook-httpx.py
hiddenimports = [
    'httpcore',
    'httpx._config',
    'httpx._content',
    'httpx._exceptions',
    'httpx._models',
    'httpx._status_codes',
    'httpx._transports.asgi',
    'httpx._transports.wsgi',
    'httpx._types',
    'httpx._urls',
    'httpx._utils',
]

# Evita que PyInstaller analice el código problemático
excludedimports = [
    'httpx._client',
    'httpx._client.AsyncClient',
    'httpx._client.Client',
]