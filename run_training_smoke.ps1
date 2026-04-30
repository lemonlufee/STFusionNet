$ErrorActionPreference = 'Stop'
python -m training.train_main --mode train --models stgcn_fusion --stf_mode default --no_tune --separate_horizons --horizon_hours 12,24,48,120,168 --top_k_lakes 4 --min_effective_steps 120 --seq_len 12 --batch_size 16 --max_epochs 1 --exp_root Training_time_log\smoke_training
