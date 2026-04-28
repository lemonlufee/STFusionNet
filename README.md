# STFusionNet Training Pipeline

This repository contains the training, evaluation, and paper-figure pipeline for multi-station water-quality forecasting with STFusionNet and baseline models.

## Included Components

- Data loading and preprocessing for station-based water-quality time series.
- Baseline models: CNN, TCN, LSTM, iTransformer, PatchTST, STGCN, and DCRNN.
- STFusionNet training and evaluation.
- Ablation experiments for the main architecture components.
- Graph-parameter sensitivity experiments for adjacency construction.
- Paper-figure generation from saved training metrics and optional inference artifacts.

## Data Availability

The original CSV data used in the paper are not included in this repository. To reproduce the pipeline, prepare a CSV file with the schema described in `docs/data_schema.md` and place it in the repository root as:

```text
processed_taihu_data_with_coords.csv
```

You may also change `RAW_DATA_FILE` in `config/config_taihu.py` to point to your own file.

## Environment Setup

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate stfusionnet
```

On managed Linux servers, load the server-provided Anaconda module first if required, then run the same commands above. After activation, verify the environment:

```bash
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA:', torch.version.cuda)"
```

If conda is unavailable, use `requirements.txt` with a local virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The PyTorch package should be compatible with the target machine. If your server already provides a working Anaconda/PyTorch stack, using that stack directly is recommended.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Minimal Full-Pipeline Check

The `quick` mode is intended for validating that the full workflow can run end-to-end with a small data subset and one training epoch. It executes:

- environment checks;
- training for STFusionNet and all baseline models;
- a reduced ablation run;
- graph sensitivity analysis for `k` and `sigma`;
- feature-regime diagnostics;
- metric-based paper-figure rendering.

Windows PowerShell:

```powershell
.\run_all_server.ps1 `
  -Mode quick `
  -RunTag opensource_smoke `
  -ExpRoot Training_time_log `
  -AblationRoot ablation_results `
  -SkipPack
```

Linux:

```bash
bash run_all_server.sh \
  --mode quick \
  --run-tag opensource_smoke \
  --exp-root Training_time_log \
  --ablation-root ablation_results \
  --skip-pack
```

If Python is not discoverable from `PATH`, pass it explicitly.

Windows PowerShell:

```powershell
.\run_all_server.ps1 -Mode quick -PythonPath C:\path\to\python.exe -SkipPack
```

Linux:

```bash
bash run_all_server.sh --mode quick --python /path/to/python --skip-pack
```

## Single-Model Smoke Test

Use this command when you only need to check that the core training entry point works.

Windows PowerShell:

```powershell
python -m training.train_main --mode train --models stgcn_fusion --stf_mode default --no_tune --top_k_lakes 4 --min_effective_steps 120 --seq_len 12 --pred_len 1 --batch_size 16 --max_epochs 1 --exp_root Training_time_log\smoke_training
```

Linux:

```bash
python -m training.train_main --mode train --models stgcn_fusion --stf_mode default --no_tune --top_k_lakes 4 --min_effective_steps 120 --seq_len 12 --pred_len 1 --batch_size 16 --max_epochs 1 --exp_root Training_time_log/smoke_training
```

## Full Experiment Run

Use `full` mode for the complete training, ablation, sensitivity, diagnostics, visualization, and packaging workflow.

Windows PowerShell:

```powershell
.\run_all_server.ps1 -Mode full
```

Linux:

```bash
bash run_all_server.sh --mode full
```

The full mode uses the complete model set, full ablation variants, full sensitivity settings, and tuning configuration defined in the pipeline scripts.

## Figure Generation Notes

The pipeline renders metric-based figures from the latest `test_metrics.json` found under the selected experiment root.

Sequence and predicted-versus-observed scatter figures require the matching `analysis_data.npz` file from the same run directory. In `quick` mode, the server pipeline intentionally uses `--no_post` to keep the validation run small, so `analysis_data.npz` may be absent and those inference-detail figures will be skipped with a warning. This warning is expected for the minimal check and does not indicate a failed run.

For paper reporting, figures and tables should be generated from a consistent metric source:

- same station set;
- same target variable;
- same prediction horizon;
- same `test_metrics.json`;
- same `analysis_data.npz` when inference-detail figures are used.

## Outputs

Generated outputs are intentionally ignored by Git.

- Training metrics, checkpoints, and run logs are written under `Training_time_log/`.
- Ablation and sensitivity outputs are written under `ablation_results/`.
- Packed artifacts are created only when packaging is enabled.

Before publishing or archiving the repository, keep only source code, configuration files, documentation, and environment files. Do not commit raw data, trained weights, generated figures, run logs, archives, or Python cache directories.

## Reproducibility Notes

- The test split is used only after training and model selection.
- Validation metrics drive learning-rate scheduling and early stopping.
- STFusionNet and all baseline models are evaluated under the same data split and metric definitions.
- The default metric family is NSE, RMSE, and MAE.
- Reported metrics should remain self-consistent across figures whenever the station, target variable, and prediction horizon are identical.
