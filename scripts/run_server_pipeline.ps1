param(
    [string]$PythonPath = "",
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

Set-StrictMode -Version Latest

function Write-Section([string]$Title) {
    Write-Host ""
    Write-Host "==================== $Title ===================="
}

function Invoke-Step([string]$Name, [scriptblock]$Cmd) {
    Write-Host "[RUN] $Name"
    & $Cmd
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $Name (exit code=$LASTEXITCODE)"
    }
    Write-Host "[OK ] $Name"
}

function Resolve-Python([string]$UserPath) {
    if (-not [string]::IsNullOrWhiteSpace($UserPath) -and (Test-Path $UserPath)) {
        return (Resolve-Path $UserPath).Path
    }

    if (-not [string]::IsNullOrWhiteSpace($env:PYTHON_EXE) -and (Test-Path $env:PYTHON_EXE)) {
        return (Resolve-Path $env:PYTHON_EXE).Path
    }

    $userHome = $env:USERPROFILE
    $candidates = @(    )
    foreach ($c in $candidates) {
        if (Test-Path $c) {
            return (Resolve-Path $c).Path
        }
    }

    $pyCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($null -ne $pyCmd -and (Test-Path $pyCmd.Source)) {
        return $pyCmd.Source
    }

    throw "No usable python found. Please pass -PythonPath explicitly."
}

# Resolve project root as script's parent parent (scripts/)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
Set-Location $ProjectRoot

$PythonPath = Resolve-Python $PythonPath

$Tag = $RunTag

Write-Section "Environment Check"
Invoke-Step "Python version" { & $PythonPath -V }
Invoke-Step "Torch/CUDA check" { & $PythonPath -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)" }

$modelList = "stgcn_fusion,cnn,tcn,lstm,itransformer,patchtst,stgcn,dcrnn"
$ablationVariants = "full,w_o_adaptive_adj,temporal_cnn_only,temporal_lstm_only,temporal_tcn_only,fusion_avg,fusion_concat"
$sensK = "3,6,10,15"
$sensSigma = "10,20,30"
$ablationEpochs = 50
$sensitivityEpochs = 50
$dataArgs = @()
$trainHorizonArgs = @("--separate_horizons", "--horizon_hours", "12,24,48,120,168")
$trainTuneArgs = @("--tune", "--stf_mode", "search", "--search_method", "grid", "--trials", "48")

$ablationTuneArgs = @("--tune", "--search_method", "grid", "--trials", "48", "--separate_horizons", "--horizon_hours", "12,24,48,120,168")
$sensitivityTuneArgs = @("--tune", "--search_method", "grid", "--trials", "48", "--separate_horizons", "--horizon_hours", "12,24,48,120,168")

if (-not $SkipTrain) {
    Write-Section "Training"
    Invoke-Step "Main training pipeline" {
        $argsList = @(
            "-m", "training.train_main",
            "--mode", "train",
            "--models", $modelList,
            "--objective", "val_nse",
            "--exp_root", $ExpRoot,
            "--tag", $Tag,
            "--no_post",
            "--no_plot_loss"
        ) + $trainHorizonArgs + $trainTuneArgs + $dataArgs
        & $PythonPath @argsList
    }
}

if (-not $SkipAblation) {
    Write-Section "Ablation"
    Invoke-Step "Ablation experiments" {
        $argsList = @(
            "-m", "experiments.exp_ablation",
            "--variants", $ablationVariants,
            "--max_epochs", "$ablationEpochs",
            "--results_root", $AblationRoot,
            "--seed", "2025"
        ) + $ablationTuneArgs + $dataArgs
        & $PythonPath @argsList
    }
}

if (-not $SkipSensitivity) {
    Write-Section "Graph Sensitivity"
    Invoke-Step "Sensitivity experiments (k, sigma)" {
        $argsList = @(
            "-m", "experiments.exp_sensitivity",
            "--k_values", $sensK,
            "--sigma_values", $sensSigma,
            "--max_epochs", "$sensitivityEpochs",
            "--exp_root", $AblationRoot,
            "--tag", "${Tag}_graph_sens"
        ) + $sensitivityTuneArgs + $dataArgs
        & $PythonPath @argsList
    }
}

if (-not $SkipRegime) {
    Write-Section "Feature Regime Report"
    Invoke-Step "Feature regime diagnostics" {
        & $PythonPath -m evaluation.eval_feature_regime `
            --out_dir $ExpRoot
    }
}

if (-not $SkipViz) {
    Write-Section "Visualization"
    $vizDir = $ExpRoot

    $summaryPath = Join-Path $ExpRoot "${Tag}_summary.json"
    $preferredTestMetric = Get-ChildItem -Path $ExpRoot -Recurse -Filter "test_metrics.json" -ErrorAction SilentlyContinue |
        Where-Object { $_.Directory.Name -like "*stgcn_fusion*_${Tag}_h12h" -or $_.Directory.Name -like "*stgcn_fusion*_h12h" } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1

    $latestTestMetric = $preferredTestMetric
    if ($null -eq $latestTestMetric) {
        $latestTestMetric = Get-ChildItem -Path $ExpRoot -Recurse -Filter "test_metrics.json" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    }

    if ($null -ne $latestTestMetric) {
        $metricsPath = $latestTestMetric.FullName
        $analysisPath = Join-Path $latestTestMetric.Directory.FullName "analysis_data.npz"
        $latestAblation = Get-ChildItem -Path $AblationRoot -Recurse -Filter "ablation_results.json" -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1
        Invoke-Step "Render thesis figures from metrics" {
            $argsList = @(
                "-m", "visualization.viz_paper_figures",
                "--test_metrics", $metricsPath,
                "--plot_horizon_hours", "12",
                "--out_dir", $vizDir
            )
            if (Test-Path $summaryPath) {
                $argsList += @("--summary_json", $summaryPath)
            }
            if ($null -ne $latestAblation) {
                $argsList += @("--ablation_results", $latestAblation.FullName)
            }
            if (Test-Path $analysisPath) {
                $argsList += @("--analysis_npz", $analysisPath)
            }
            & $PythonPath @argsList
        }
    } else {
        Write-Host "[WARN] No test_metrics.json found. Skip inference visualizations."
    }

}

if (-not $SkipPack) {
    Write-Section "Pack Artifacts"
    $bundleName = "server_artifacts_${Tag}.zip"
    $bundlePath = Join-Path $ProjectRoot $bundleName
    if (Test-Path $bundlePath) {
        Remove-Item -LiteralPath $bundlePath -Force
    }

    $packTargets = @()
    if (Test-Path $ExpRoot) { $packTargets += (Resolve-Path $ExpRoot).Path }
    if (Test-Path $AblationRoot) { $packTargets += (Resolve-Path $AblationRoot).Path }
    
    if ($packTargets.Count -gt 0) {
        Compress-Archive -Path $packTargets -DestinationPath $bundlePath -Force
        Write-Host "[OK ] Artifacts packed: $bundlePath"
    } else {
        Write-Host "[WARN] Nothing to pack."
    }
}

Write-Section "Done"
Write-Host "All requested steps completed."
Write-Host "Project root: $ProjectRoot"
Write-Host "Python: $PythonPath"
Write-Host "Run tag: $Tag"
