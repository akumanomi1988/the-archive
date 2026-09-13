# The Archive para .NET

Port independiente de la aplicación Python a ASP.NET Core 8. Conserva las URLs principales, el formato de configuración y el esquema SQLite para poder reutilizar una biblioteca existente.

## Arranque

En Windows, ejecuta `run.cmd` o `run.ps1`. También puedes usar:

```powershell
dotnet run
```

El servidor escucha en `http://localhost:8000` y abre la web en el navegador predeterminado. Usa `run.cmd --no-open` para evitar que se abra. El puerto y este comportamiento se pueden cambiar con `port` y `open_browser` en `config.json`.

La primera ejecución crea `biblioteca`, `skald.db` y sus tablas cuando todavía no existen. Pulsa **Reindexar** en la interfaz para importar los EPUB.

## Navegadores antiguos

`http://localhost:8000/legacy` ofrece una interfaz deliberadamente sencilla: HTML clásico, CSS básico y XMLHttpRequest compatible con JavaScript antiguo. `/ebook` redirige a esa vista. Las descargas en `/ebook/raw/{id}.epub` soportan peticiones HTTP Range.

## Configuración y datos existentes

Las claves principales de `config.json` coinciden con la versión Python. Las rutas relativas se resuelven desde la carpeta del archivo de configuración. `SKALD_CONFIG` permite elegir otro archivo.

Para compartir la base anterior, establece `db_path` con la ruta de su `skald.db` y `library_path` con la biblioteca original. El port usa las tablas `book`, `bookstate`, `author`, `genre`, `bookauthor` y `bookgenre` del proyecto Python.

## Ejecutable autónomo para Windows

```powershell
.\publish-windows.ps1
```

El resultado queda en `publish\TheArchive.exe` e incluye el runtime de .NET. Copia junto al ejecutable `config.json` y la carpeta `wwwroot` generada por `dotnet publish`.

## Funciones portadas

- búsqueda, filtros, detalle y paginación;
- indexación recursiva de EPUB y lectura de metadatos OPF;
- lectura ligera del contenido, portada y descarga con HTTP Range;
- favoritos, leído, pendiente y progreso;
- configuración editable y recargable;
- enriquecimiento automático con la API pública de Open Library, sin claves ni cuentas;
- endpoints de lote y seguimiento del enriquecimiento.

## Metadatos de Open Library

El botón **Enriquecer** consulta Open Library por título y autor y guarda año, género, materias, idiomas, valoración, número de ediciones, identificador, enlace y portada disponibles. No requiere ninguna clave API. Las consultas se serializan y aplican una pausa configurable (`metadata_throttle_ms`) para mantener un uso de bajo volumen del servicio público.
