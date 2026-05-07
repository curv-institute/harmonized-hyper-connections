#!/usr/bin/env bash
set -euo pipefail

MODES=(hc mhc harm)
SEEDS=(0 1 2)
STEPS="${STEPS:-15000}"
LOG_EVERY="${LOG_EVERY:-100}"
DEVICE="${DEVICE:-auto}"
GAIN_TARGET="${GAIN_TARGET:-12.0}"
MIN_SCALE="${MIN_SCALE:-0.02}"
HARM_K="${HARM_K:-0.5}"
BETA="${BETA:-0.95}"
EXECUTION_MODE="${EXECUTION_MODE:-local}"
LOCAL_UV="${LOCAL_UV:-uv}"
REMOTE_UV="${REMOTE_UV:-/home/jwm/.local/bin/uv}"
REMOTE_ENV="${REMOTE_ENV:-/home/jwm/.cache/harmonized-hyper-connections/.venv-gx10-e313}"
LOCAL_GPU="${LOCAL_GPU:-0}"
REMOTE_GPU="${REMOTE_GPU:-0}"

cd "$(dirname "$0")/.."

run_local() {
  local mode="$1"
  local seed="$2"
  local out="runs/pub15k_${mode}_seed${seed}"
  mkdir -p "${out}"
  local cmd="CUDA_VISIBLE_DEVICES=${LOCAL_GPU} ${LOCAL_UV} run python harmonizer_applied_gain.py --mode ${mode} --steps ${STEPS} --seed ${seed} --out-dir ${out} --device ${DEVICE} --log-every ${LOG_EVERY} --gain_target ${GAIN_TARGET} --min_scale ${MIN_SCALE} --harm_k ${HARM_K} --beta ${BETA}"
  printf '%s\n' "${cmd}" > "${out}/command.txt"
  echo "Running local ${out}"
  bash -lc "${cmd}" > "${out}/stdout.log" 2> "${out}/stderr.log"
}

run_remote() {
  local mode="$1"
  local seed="$2"
  local out="runs/pub15k_${mode}_seed${seed}"
  mkdir -p "${out}"
  local remote_inner="cd /gfs/git/harmonized-hyper-connections && CUDA_VISIBLE_DEVICES=${REMOTE_GPU} UV_PROJECT_ENVIRONMENT=${REMOTE_ENV} ${REMOTE_UV} run python harmonizer_applied_gain.py --mode ${mode} --steps ${STEPS} --seed ${seed} --out-dir ${out} --device ${DEVICE} --log-every ${LOG_EVERY} --gain_target ${GAIN_TARGET} --min_scale ${MIN_SCALE} --harm_k ${HARM_K} --beta ${BETA}"
  local cmd="ssh gx10-e313 '${remote_inner}'"
  printf '%s\n' "${cmd}" > "${out}/command.txt"
  echo "Running remote ${out}"
  ssh -o BatchMode=yes gx10-e313 "${remote_inner}" > "${out}/stdout.log" 2> "${out}/stderr.log"
}

should_skip() {
  local mode="$1"
  local seed="$2"
  local out="runs/pub15k_${mode}_seed${seed}"
  [[ -f "${out}/summary.json" && -f "${out}/metrics.csv" && -f "${out}/manifest.json" ]]
}

run_one() {
  local target="$1"
  local mode="$2"
  local seed="$3"
  if should_skip "${mode}" "${seed}"; then
    echo "Skipping existing runs/pub15k_${mode}_seed${seed}"
    return 0
  fi
  if [[ "${target}" == "remote" ]]; then
    run_remote "${mode}" "${seed}"
  else
    run_local "${mode}" "${seed}"
  fi
}

if [[ "${EXECUTION_MODE}" == "cluster" ]]; then
  jobs=()
  for mode in "${MODES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      jobs+=("${mode}:${seed}")
    done
  done

  i=0
  while [[ "${i}" -lt "${#jobs[@]}" ]]; do
    IFS=: read -r mode seed <<< "${jobs[$i]}"
    run_one local "${mode}" "${seed}" &
    pid_local=$!
    i=$((i + 1))

    pid_remote=""
    if [[ "${i}" -lt "${#jobs[@]}" ]]; then
      IFS=: read -r mode seed <<< "${jobs[$i]}"
      run_one remote "${mode}" "${seed}" &
      pid_remote=$!
      i=$((i + 1))
    fi

    status=0
    wait "${pid_local}" || status=$?
    if [[ -n "${pid_remote}" ]]; then
      wait "${pid_remote}" || status=$?
    fi
    if [[ "${status}" -ne 0 ]]; then
      exit "${status}"
    fi
  done
else
  for mode in "${MODES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      run_one local "${mode}" "${seed}"
    done
  done
fi

"${LOCAL_UV}" run python scripts/validate_artifacts.py --prefix pub15k --steps "${STEPS}"
