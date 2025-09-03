# gunicorn.conf.py
import multiprocessing

bind = "0.0.0.0:8000"  # Dirección y puerto
workers = multiprocessing.cpu_count() * 2 + 1  # Workers basados en núcleos de CPU
worker_class = "uvicorn.workers.UvicornWorker"  # Usa Uvicorn como worker
timeout = 120  # Tiempo máximo de espera por solicitud (en segundos)
keepalive = 5  # Tiempo de vida de conexiones persistentes (en segundos)
loglevel = "info"  # Nivel de logging
accesslog = "-"  # Log de acceso a stdout
errorlog = "-"  # Log de errores a stdout