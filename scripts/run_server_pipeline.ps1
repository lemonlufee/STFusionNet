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

    if (-not (Test-Path $summaryPath)) {
        Write-Host "[WARN] Summary JSON not found: $summaryPath. Skip visualization."
    } else {
        # Resolve STFusionNet 12h run_dir from the summary JSON rather than
        # picking the newest file by mtime, so an older run directory left
        # behind from a previous pipeline does not hijack the figures.
        $stfRunDir = ""
        try {
            $summaryRaw = Get-Content -LiteralPath $summaryPath -Raw -Encoding UTF8
            $summaryObj = $summaryRaw | ConvertFrom-Json
            $resultsList = @()
            if ($null -ne $summaryObj.results) {
                $resultsList = @($summaryObj.results)
            }
            $stfNames = @("stgcn_fusion", "stfusionnet")
            foreach ($item in $resultsList) {
                if ($null -eq $item) { continue }
                $modelNameRaw = ""
                if ($null -ne $item.model) { $modelNameRaw = [string]$item.model }
                $modelName = $modelNameRaw.Trim().ToLower()
                if ($stfNames -notcontains $modelName) { continue }
                if ($null -eq $item.horizon_hours) { continue }
                $horizonVal = 0
                if (-not [int]::TryParse([string]$item.horizon_hours, [ref]$horizonVal)) { continue }
                if ($horizonVal -ne 12) { continue }
                if ($null -ne $item.run_dir -and -not [string]::IsNullOrWhiteSpace([string]$item.run_dir)) {
                    $stfRunDir = [string]$item.run_dir
                    break
                }
            }
        } catch {
            Write-Host "[WARN] Failed to parse summary JSON: $($_.Exception.Message)"
        }

        if ([string]::IsNullOrWhiteSpace($stfRunDir)) {
            Write-Host "[WARN] $summaryPath has no STFusionNet 12h entry with run_dir. Skip visualization."
        } elseif (-not (Test-Path (Join-Path $stfRunDir "test_metrics.json"))) {
            Write-Host "[WARN] STFusionNet 12h run_dir '$stfRunDir' has no test_metrics.json. Skip visualization."
        } else {
            $stfTestMetrics = Join-Path $stfRunDir "test_metrics.json"
            $stfAnalysisNpz = Join-Path $stfRunDir "analysis_data.npz"
            $latestAblation = Get-ChildItem -Path $AblationRoot -Recurse -Filter "ablation_results.json" -ErrorAction SilentlyContinue |
                Sort-Object LastWriteTime -Descending |
                Select-Object -First 1

            Invoke-Step "Render thesis figures from metrics" {
                $argsList = @(
                    "-m", "visualization.viz_paper_figures",
                    "--summary_json", $summaryPath,
                    "--test_metrics", $stfTestMetrics,
                    "--plot_horizon_hours", "12",
                    "--out_dir", $vizDir
                )
                if (Test-Path $stfAnalysisNpz) {
                    $argsList += @("--analysis_npz", $stfAnalysisNpz)
                } else {
                    Write-Host "[WARN] STFusionNet 12h run_dir '$stfRunDir' has no analysis_data.npz; sequence/scatter figures may be incomplete."
                }
                if ($null -ne $latestAblation) {
                    $argsList += @("--ablation_results", $latestAblation.FullName)
                }
                & $PythonPath @argsList
            }
        }
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
