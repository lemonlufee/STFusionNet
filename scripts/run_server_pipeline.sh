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

  if [[ ! -f "${summary_json}" ]]; then
    echo "[WARN] Summary JSON not found: ${summary_json}. Skip visualization."
  else
    # Resolve STFusionNet 12h run_dir from the summary JSON rather than
    # picking the newest file by mtime, so an older run directory left
    # behind from a previous pipeline does not hijack the figures.
    stf_run_dir="$(
      "${PYTHON_PATH}" - "${summary_json}" <<'PY'
import json, sys
try:
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        data = json.load(f)
except Exception:
    sys.exit(0)
targets = {"stgcn_fusion", "stfusionnet"}
matches = []
for r in (data.get("results") or []):
    if not isinstance(r, dict):
        continue
    name = str(r.get("model", "")).strip().lower()
    if name not in targets:
        continue
    horizon = r.get("horizon_hours")
    if horizon is None:
        continue
    try:
        if int(horizon) == 12:
            matches.append(r)
    except Exception:
        continue
if not matches:
    sys.exit(0)
run_dir = matches[0].get("run_dir") or ""
if run_dir:
    print(run_dir)
PY
)"
    stf_run_dir="${stf_run_dir//$'\r'/}"

    if [[ -z "${stf_run_dir}" ]]; then
      echo "[WARN] ${summary_json} has no STFusionNet 12h entry with run_dir. Skip visualization."
    elif [[ ! -f "${stf_run_dir}/test_metrics.json" ]]; then
      echo "[WARN] STFusionNet 12h run_dir '${stf_run_dir}' has no test_metrics.json. Skip visualization."
    else
      stf_test_metrics="${stf_run_dir}/test_metrics.json"
      stf_analysis_npz="${stf_run_dir}/analysis_data.npz"
      latest_ablation="$(find "${ABLATION_ROOT}" -type f -name ablation_results.json 2>/dev/null | xargs -r ls -t | head -n 1 || true)"

      viz_args=(
        -m visualization.viz_paper_figures
        --summary_json "${summary_json}"
        --test_metrics "${stf_test_metrics}"
        --plot_horizon_hours 12
        --out_dir "${VIZ_DIR}"
      )
      if [[ -f "${stf_analysis_npz}" ]]; then
        viz_args+=(--analysis_npz "${stf_analysis_npz}")
      else
        echo "[WARN] STFusionNet 12h run_dir '${stf_run_dir}' has no analysis_data.npz; sequence/scatter figures may be incomplete."
      fi
      if [[ -n "${latest_ablation}" && -f "${latest_ablation}" ]]; then
        viz_args+=(--ablation_results "${latest_ablation}")
      fi

      run_step "Render thesis figures from metrics" "${PYTHON_PATH}" "${viz_args[@]}"
    fi
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
