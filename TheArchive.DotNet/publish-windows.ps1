$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
dotnet publish .\TheArchive.DotNet.csproj -c Release -r win-x64 --self-contained true -p:PublishSingleFile=true -o .\publish
Copy-Item -LiteralPath .\config.json -Destination .\publish\config.json -Force
Write-Host 'Generado en publish\TheArchive.exe'
