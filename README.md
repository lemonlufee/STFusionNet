# STFusionNet Training Pipeline

This repository contains the training, evaluation, and paper-figure pipeline for multi-station water-quality forecasting with STFusionNet and baseline models.

## Included components

- Data loading and preprocessing for station-based water-quality time series.
- Baseline models: CNN, TCN, LSTM, iTransformer, PatchTST, STGCN, and DCRNN.
- STFusionNet training and evaluation.
- Ablation and graph-parameter sensitivity experiment entry points.
- Inference-only paper figure generation from saved training outputs.

## Data availability

The original CSV data used in the paper are not included in this repository. To reproduce the pipeline, prepare a CSV file with the schema described in `docs/data_schema.md` and place it in the project root as:

```text
processed_taihu_data_with_coords.csv
```

You may also change `RAW_DATA_FILE` in `config/config_taihu.py` to point to your own file.

## Quick smoke test

Windows PowerShell:

```powershell
python -m training.train_main --mode train --models stgcn_fusion --stf_mode default --no_tune --top_k_lakes 4 --min_effective_steps 120 --seq_len 12 --pred_len 1 --batch_size 16 --max_epochs 1 --exp_root Training_time_log\smoke_training
```

Linux:

```bash
python -m training.train_main --mode train --models stgcn_fusion --stf_mode default --no_tune --top_k_lakes 4 --min_effective_steps 120 --seq_len 12 --pred_len 1 --batch_size 16 --max_epochs 1 --exp_root Training_time_log/smoke_training
```

## Full server pipeline

Windows PowerShell:

```powershell
.\run_all_server.ps1 -Mode quick
```

Linux:

```bash
bash run_all_server.sh --mode quick
```

Use `full` mode for the full training, ablation, sensitivity, diagnostics, and figure-generation pipeline.

## Outputs

Training outputs are written under `Training_time_log/`. Thesis figures are stored in the current run directory when post-processing is enabled. Generated outputs are intentionally ignored by Git.

## Reproducibility notes

- The test split is used only after training and model selection.
- Validation metrics drive learning-rate scheduling and early stopping.
- Reported figure metrics should be generated from the same `test_metrics.json` and `analysis_data.npz` pair.
