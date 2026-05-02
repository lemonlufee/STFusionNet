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

## Full Experiment Run

This open-source tree exposes only the full training pipeline.

The full workflow runs training, ablation, graph-parameter sensitivity analysis, feature-regime diagnostics, visualization, and artifact packaging.

Windows PowerShell:

```powershell
.
un_all_server.ps1
```

Linux:

```bash
bash run_all_server.sh
```

If Python is not discoverable from `PATH`, pass it explicitly.

Windows PowerShell:

```powershell
.
un_all_server.ps1 -PythonPath C:\path	o\python.exe
```

Linux:

```bash
bash run_all_server.sh --python /path/to/python
```

The full pipeline uses the complete model set, full ablation variants, full sensitivity settings, and horizon-specific training. Main model comparison is trained as independent forecast-horizon runs for `12h`, `24h`, `48h`, `120h`, and `168h`; each horizon has its own `PRED_LEN`, run directory, and metric record. The five forecast horizons are part of the same Full-model experiment design: they are not separate ablation variants.

The pipeline enables paper-style hyperparameter search for the main model comparison, ablation variants, and graph sensitivity runs. Each model/variant/graph-setting is tuned separately at each required forecast horizon using the same compact grid:

- `SEQ_LEN`: `18`, `30`, `42`;
- hidden size: `32`, `64`, `128`, `256`;
- learning rate: `5e-5`, `1e-4`, `3e-4`, `1e-3`.

This gives `48` tuning trials plus one final training run for each model/variant/graph-setting at each horizon. With the default full configuration, the expected number of training executions is `6615`: `1960` for model comparison, `1715` for ablation, and `2940` for graph-parameter sensitivity.

## Figure Generation Notes

The pipeline renders metric-based figures from the latest `test_metrics.json` found under the selected experiment root.

Sequence and predicted-versus-observed scatter figures require the matching `analysis_data.npz` file from the same run directory. Use the same run directory for figure rendering and metric reporting to keep values self-consistent.

For paper reporting, figures and tables should be generated from a consistent metric source:

- same station set;
- same target variable;
- same prediction horizon;
- same `test_metrics.json`;
- same `analysis_data.npz` when inference-detail figures are used.

The main comparison figures use horizon-specific runs. For example, `12h` sequence and scatter figures are computed from the final step of the `12h` run, not from an average over multiple future steps.

## Outputs

Generated outputs are intentionally ignored by Git.

- Training metrics, checkpoints, and run logs are written under `Training_time_log/`.
- Ablation and sensitivity outputs are written under `ablation_results/`.
- Packed artifacts are created only when packaging is enabled.

The main paper-style figures are written into the selected experiment root:

- `nse_panels_custom.png`: STFusionNet NSE panels across the five required horizons.
- `water_quality_nse_linear_fit.png`: NSE-horizon curves for STFusionNet and baseline models.
- `image.png`: baseline comparison bars for NSE and RMSE.
- `yuceshixu.png`: 12h prediction sequence panels from the selected STFusionNet run.
- `scatter_custom.png`: 12h predicted-versus-observed scatter panels from the same run.
- `ablation_module_matrix.png`: ablation module ON/OFF matrix.
- `ablation_nse_panels.png`: ablation NSE panels.

Before publishing or archiving the repository, keep only source code, configuration files, documentation, and environment files. Do not commit raw data, trained weights, generated figures, run logs, archives, or Python cache directories.

## Reproducibility Notes

- The test split is used only after training and model selection.
- Validation metrics drive learning-rate scheduling and early stopping.
- STFusionNet and all baseline models are evaluated under the same data split and metric definitions.
- The default metric family is NSE, RMSE, and MAE.
- Reported metrics should remain self-consistent across figures whenever the station, target variable, and prediction horizon are identical.
