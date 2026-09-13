$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
dotnet run --project .\TheArchive.DotNet.csproj -- @args
