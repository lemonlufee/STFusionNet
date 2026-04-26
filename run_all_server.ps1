param(
    [string]$PythonPath = "",
    [ValidateSet("full", "quick")]
    [string]$Mode = "full",
    [string]$RunTag = "server"
)

$ErrorActionPreference = "Stop"
$script = Join-Path $PSScriptRoot "scripts\run_server_pipeline.ps1"
if (-not (Test-Path $script)) {
    throw "Cannot find pipeline script: $script"
}
powershell -ExecutionPolicy Bypass -File $script -Mode $Mode -PythonPath $PythonPath -RunTag $RunTag
