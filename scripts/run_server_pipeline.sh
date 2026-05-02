#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
#   bash scripts/run_server_pipeline.sh
#   bash scripts/run_server_pipeline.sh --python /path/to/python

RUN_TAG="server"
EXP_ROOT="Training_time_log"
ABLATION_ROOT="ablation_results"
PYTHON_PATH=""
SKIP_TRAIN=0
SKIP_ABLATION=0
SKIP_SENS=0
SKIP_REGIME=0
SKIP_VIZ=0
SKIP_PACK=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) PYTHON_PATH="$2"; shift 2 ;;
    --run-tag) RUN_TAG="$2"; shift 2 ;;
    --exp-root) EXP_ROOT="$2"; shift 2 ;;
    --ablation-root) ABLATION_ROOT="$2"; shift 2 ;;
    --skip-train) SKIP_TRAIN=1; shift ;;
    --skip-ablation) SKIP_ABLATION=1; shift ;;
    --skip-sensitivity) SKIP_SENS=1; shift ;;
    --skip-regime) SKIP_REGIME=1; shift ;;
    --skip-viz) SKIP_VIZ=1; shift ;;
    --skip-pack) SKIP_PACK=1; shift ;;
    *) echo "Unknown arg: $1"; exit 2 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ -z "${PYTHON_PATH}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_PATH="$(command -v python)"
  else
    echo "No python found in PATH. Pass --python /path/to/python"
    exit 1
  fi
fi

sanitize_ld_library_path() {
  # Prefer the active environment's lib directory and remove duplicate
  # entries. This keeps managed-server shared libraries from shadowing the
  # environment selected by --python without hard-coding any institution path.
  local old_ld="${LD_LIBRARY_PATH:-}"
  local IFS=':'
  local arr=($old_ld)
  local cleaned=()
  for p in "${arr[@]}"; do
    [[ -z "${p}" ]] && continue
    cleaned+=("${p}")
  done

  local py_bin py_dir env_dir env_lib
  py_bin="$(readlink -f "${PYTHON_PATH}" || echo "${PYTHON_PATH}")"
  py_dir="$(dirname "${py_bin}")"
  env_dir="$(dirname "${py_dir}")"
  env_lib="${env_dir}/lib"

  if [[ -d "${env_lib}" ]]; then
    local deduped=()
    for p in "${cleaned[@]}"; do
      [[ "${p}" == "${env_lib}" ]] && continue
      deduped+=("${p}")
    done
    if [[ ${#deduped[@]} -eq 0 ]]; then
      export LD_LIBRARY_PATH="${env_lib}"
    else
      export LD_LIBRARY_PATH="${env_lib}:$(IFS=:; echo "${deduped[*]}")"
    fi
  else
    if [[ ${#cleaned[@]} -eq 0 ]]; then
      unset LD_LIBRARY_PATH
    else
      export LD_LIBRARY_PATH="$(IFS=:; echo "${cleaned[*]}")"
    fi
  fi
}

sanitize_ld_library_path

TAG="${RUN_TAG}"

run_step() {
  local name="$1"
  shift
  echo "[RUN] ${name}"
  "$@"
  echo "[OK ] ${name}"
}

echo "==================== Environment Check ===================="
run_step "Python version" "${PYTHON_PATH}" -V
run_step "Torch/CUDA check" "${PYTHON_PATH}" -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
if "${PYTHON_PATH}" -c "import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  run_step "Torch cuDNN runtime check" "${PYTHON_PATH}" -c "import torch; x=torch.randn(2,3,16,16,device='cuda'); m=torch.nn.Conv2d(3,8,3,padding=1).cuda(); y=m(x); print('ok', tuple(y.shape))"
else
  run_step "Torch CPU runtime check" "${PYTHON_PATH}" -c "import torch; x=torch.randn(2,3,16,16); m=torch.nn.Conv2d(3,8,3,padding=1); y=m(x); print('ok', tuple(y.shape))"
fi

MODEL_LIST="stgcn_fusion,cnn,tcn,lstm,itransformer,patchtst,stgcn,dcrnn"
ABLATION_VARIANTS="full,w_o_adaptive_adj,temporal_cnn_only,temporal_lstm_only,temporal_tcn_only,fusion_avg,fusion_concat"
SENS_K="3,6,10,15"
SENS_SIGMA="10,20,30"
ABLATION_EPOCHS=50
SENS_EPOCHS=50
DATA_ARGS=()
TRAIN_HORIZON_ARGS=(--separate_horizons --horizon_hours 12,24,48,120,168)
TRAIN_TUNE_ARGS=(--tune --stf_mode search --search_method grid --trials 48)
ABLATION_TUNE_ARGS=(--tune --search_method grid --trials 48 --separate_horizons --horizon_hours 12,24,48,120,168)
SENS_TUNE_ARGS=(--tune --search_method grid --trials 48 --separate_horizons --horizon_hours 12,24,48,120,168)


if [[ ${SKIP_TRAIN} -eq 0 ]]; then
  echo "==================== Training ===================="
run_step "Main training pipeline" \
    "${PYTHON_PATH}" -m training.train_main \
      --mode train \
      --models "${MODEL_LIST}" \
      --objective val_nse \
      --exp_root "${EXP_ROOT}" \
      --tag "${TAG}" \
      --no_post \
      --no_plot_loss \
      "${TRAIN_HORIZON_ARGS[@]}" \
      "${TRAIN_TUNE_ARGS[@]}" \
      "${DATA_ARGS[@]}"
fi

if [[ ${SKIP_ABLATION} -eq 0 ]]; then
  echo "==================== Ablation ===================="
  run_step "Ablation experiments" \
    "${PYTHON_PATH}" -m experiments.exp_ablation \
      --variants "${ABLATION_VARIANTS}" \
      --max_epochs "${ABLATION_EPOCHS}" \
      --results_root "${ABLATION_ROOT}" \
      --seed 2025 \
      "${ABLATION_TUNE_ARGS[@]}" \
      "${DATA_ARGS[@]}"
fi

if [[ ${SKIP_SENS} -eq 0 ]]; then
  echo "==================== Graph Sensitivity ===================="
  run_step "Sensitivity experiments (k, sigma)" \
    "${PYTHON_PATH}" -m experiments.exp_sensitivity \
      --k_values "${SENS_K}" \
      --sigma_values "${SENS_SIGMA}" \
      --max_epochs "${SENS_EPOCHS}" \
      --exp_root "${ABLATION_ROOT}" \
      --tag "${TAG}_graph_sens" \
      "${SENS_TUNE_ARGS[@]}" \
      "${DATA_ARGS[@]}"
fi

if [[ ${SKIP_REGIME} -eq 0 ]]; then
  echo "==================== Feature Regime ===================="
  run_step "Feature regime diagnostics" \
    "${PYTHON_PATH}" -m evaluation.eval_feature_regime \
      --out_dir "${EXP_ROOT}"
fi

if [[ ${SKIP_VIZ} -eq 0 ]]; then
  echo "==================== Visualization ===================="
  VIZ_DIR="${EXP_ROOT}"

  summary_json="${EXP_ROOT}/${TAG}_summary.json"
  preferred_test_metrics="$(find "${EXP_ROOT}" -type f -path "*stgcn_fusion*_${TAG}_h12h/test_metrics.json" 2>/dev/null | head -n 1 || true)"
  if [[ -z "${preferred_test_metrics}" ]]; then
    preferred_test_metrics="$(find "${EXP_ROOT}" -type f -path "*stgcn_fusion*_h12h/test_metrics.json" 2>/dev/null | head -n 1 || true)"
  fi
  latest_test_metrics="${preferred_test_metrics}"
  if [[ -z "${latest_test_metrics}" ]]; then
    latest_test_metrics="$(find "${EXP_ROOT}" -type f -name test_metrics.json 2>/dev/null | xargs -r ls -t | head -n 1 || true)"
  fi
  if [[ -n "${latest_test_metrics}" ]]; then
    analysis_npz="$(dirname "${latest_test_metrics}")/analysis_data.npz"
    latest_ablation="$(find "${ABLATION_ROOT}" -type f -name ablation_results.json 2>/dev/null | xargs -r ls -t | head -n 1 || true)"
    viz_args=(
      -m visualization.viz_paper_figures
      --test_metrics "${latest_test_metrics}"
      --plot_horizon_hours 12
      --out_dir "${VIZ_DIR}"
    )
    if [[ -f "${summary_json}" ]]; then
      viz_args+=(--summary_json "${summary_json}")
    fi
    if [[ -n "${latest_ablation}" && -f "${latest_ablation}" ]]; then
      viz_args+=(--ablation_results "${latest_ablation}")
    fi
    if [[ -f "${analysis_npz}" ]]; then
      viz_args+=(--analysis_npz "${analysis_npz}")
    fi
    run_step "Render thesis figures from metrics" "${PYTHON_PATH}" "${viz_args[@]}"
  else
    echo "[WARN] No test_metrics.json found; skip inference visualizations."
  fi
fi

if [[ ${SKIP_PACK} -eq 0 ]]; then
  echo "==================== Pack Artifacts ===================="
  BUNDLE="server_artifacts_${TAG}.tar.gz"
  tar -czf "${BUNDLE}" "${EXP_ROOT}" "${ABLATION_ROOT}" 2>/dev/null || tar -czf "${BUNDLE}" "${EXP_ROOT}" "${ABLATION_ROOT}" 2>/dev/null || true
  if [[ -f "${BUNDLE}" ]]; then
    echo "[OK ] Artifacts packed: ${PROJECT_ROOT}/${BUNDLE}"
  else
    echo "[WARN] Nothing packed."
  fi
fi

echo "==================== Done ===================="
echo "All requested steps completed."
echo "Project root: ${PROJECT_ROOT}"
echo "Python: ${PYTHON_PATH}"
echo "Run tag: ${TAG}"
