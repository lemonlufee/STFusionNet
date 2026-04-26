param(
    [string]$PythonPath = "",
    [ValidateSet("full", "quick")]
    [string]$Mode = "full",
    [string]$RunTag = "server",
    [string]$ExpRoot = "Training_time_log",
    [string]$AblationRoot = "ablation_results",
    [switch]$SkipTrain,
    [switch]$SkipAblation,
    [switch]$SkipSensitivity,
    [switch]$SkipRegime,
    [switch]$SkipViz,
    [switch]$SkipPack
)

$ErrorActionPreference = "Stop"
$script = Join-Path $PSScriptRoot "scripts\run_server_pipeline.ps1"
if (-not (Test-Path $script)) {
    throw "Cannot find pipeline script: $script"
}

$argsList = @(
    "-ExecutionPolicy", "Bypass",
    "-File", $script,
    "-Mode", $Mode,
    "-RunTag", $RunTag,
    "-ExpRoot", $ExpRoot,
    "-AblationRoot", $AblationRoot
)
if (-not [string]::IsNullOrWhiteSpace($PythonPath)) { $argsList += @("-PythonPath", $PythonPath) }
if ($SkipTrain) { $argsList += "-SkipTrain" }
if ($SkipAblation) { $argsList += "-SkipAblation" }
if ($SkipSensitivity) { $argsList += "-SkipSensitivity" }
if ($SkipRegime) { $argsList += "-SkipRegime" }
if ($SkipViz) { $argsList += "-SkipViz" }
if ($SkipPack) { $argsList += "-SkipPack" }

powershell @argsList
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }