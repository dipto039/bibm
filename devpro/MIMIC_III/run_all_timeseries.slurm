#!/bin/bash
# ==============================================================================
# Driver for the MIMIC-III plain 3-class (multiclass, NOT ordinal) time-series
# LOS sweep: builds the cohort + hourly dynamic features from raw MIMIC-III
# CSVs, imputes/preprocesses into model-ready tensors, trains all 6 BIBM
# architectures, and combines their test metrics into one summary CSV.
#
# Mirrors clean_LOS/run_all.slurm's structure/conventions (ACCOUNT/PARTITION
# edit block, ENV_SETUP thread-pinning, per-job sbatch submission with
# --dependency=afterok chaining through Stage 0, a final combine step on
# --dependency=afterany so a partial sweep still gets a summary) -- adapted
# for two envs (CPU data-prep, GPU training) instead of clean_LOS's
# DATAPREP_ENV/ML_ENV (both CPU) split, since this pipeline trains its own
# torch+mamba_ssm models instead of handing off to classical ML, and for a
# GPU partition on the 6 training jobs.
#
# Only the Multiclass (plain 3-way softmax, not the cumulative-logit Ordinal
# formulation) task is wired up here, per current scope -- Binary/Ordinal
# would need their own equivalent of Stage 1 below if added later.
#
# Reads raw MIMIC-III v1.4 CSVs from <project_root>/original_data (the SAME
# raw data clean_LOS already uses). Writes intermediate build/impute/
# preprocess outputs under
# <project_root>/processed_data/structured_LOS_dynamic. Writes final model
# checkpoints/metrics/summary CSV under <project_root>/results_timeseries --
# a separate results tree from clean_LOS's ../results_LOS so the two
# pipelines never collide.
#
# Usage:   bash run_all_timeseries.slurm
# (Launcher script: it calls `sbatch` itself for every job, rather than being
#  a single #SBATCH job.)
#
# Crash-safety: every Python script in this pipeline already self-locates its
# input/output paths via __file__ (see each script's _PROJECT_ROOT /
# DATA_DIR / OUT_DIR computation) -- none of them need path arguments passed
# in from this driver, unlike clean_LOS's run_all.slurm which passes
# --raw_data_dir/--out_dir explicitly because ITS scripts default to
# cwd-relative paths instead.
# ==============================================================================
# ODU HPC (Wahab) settings -- EDIT THIS BLOCK FOR YOUR ACCOUNT
# Reference: https://wiki.hpc.odu.edu/en/slurm (verify partition names
# against `sinfo` / `sacctmgr show user $USER` before your first real run --
# PARTITION_GPU="gpu" below is an unverified placeholder).
# ==============================================================================
set -euo pipefail

ACCOUNT=""                  # e.g. ACCOUNT="my_pi_lab" -- leave empty if not required
PARTITION_CPU="main"          # CPU partition: build/impute/preprocess + combine_results
PARTITION_GPU="gpu"           # GPU partition for the 6 training jobs -- VERIFY against `sinfo`
GPUS_PER_JOB="1"
IMPUTER="median"               # knn (slow at full scale -- timed out at 4h on Wahab), mean, or median
# strict (default, ~38k admissions): LOS>=1d, single-qualifying-stay, age>=18
# original (sibtanu's filters, ~50k admissions): first-stay-only, no LOS/age filter
# Change to "original" to run the comparison experiment.
COHORT_FILTER="strict"

# Wahab wraps conda in a container -- there is no `conda activate`. TWO
# separate envs, built INTERACTIVELY (never inside a job script -- see
# https://wiki.hpc.odu.edu/Software/Conda):
#
#   DATAPREP_ENV (V12.1_build_script.py / V12.1_impute_script.py /
#   V12.1_preprocess_script.py / combine_results.py -- no torch needed, the
#   latter is stdlib-only):
#     module load container_env python3/2024.2-py310
#     mkdir -p ~/envs
#     crun -c -p ~/envs/MIMIC-dataprep
#     crun -p ~/envs/MIMIC-dataprep pip install -r ../../../requirements-dataprep.txt
#
#   TRAIN_ENV (every V1_train_mimic3_*_multiclass_auc.py -- needs CUDA torch
#   + mamba_ssm; see requirements-train.txt for the full install sequence,
#   including the causal-conv1d/mamba-ssm prebuilt-wheel notes -- compiling
#   either from source does NOT work on Wahab, its "cuda/*" modules are
#   Singularity-container wrappers with no host-reachable nvcc):
#     module load container_env python3/2024.2-py310
#     crun -c -p ~/envs/MIMIC-train
#     crun -p ~/envs/MIMIC-train pip install torch --index-url https://download.pytorch.org/whl/cu121
#     crun -p ~/envs/MIMIC-train pip install -r ../requirements-train.txt
#     (+ causal-conv1d / mamba-ssm prebuilt wheels matching cp310+torch2.5+cu12+cxx11abiFALSE)
#
# IMPORTANT: never `module load cuda/*` in an interactive shell you're about
# to run `crun -p ... ` from -- it prepends a directory containing a
# DIFFERENT `crun` (for entering that module's Singularity container) ahead
# of the real one on PATH, breaking every `crun -p <venv> ...` command with
# "exec: -p: invalid option" until you `module unload cuda/*`. Not an issue
# inside this script (sbatch jobs start a fresh shell), just a footgun when
# setting up the envs above interactively.
DATAPREP_ENV="$HOME/envs/MIMIC-dataprep"
TRAIN_ENV="$HOME/envs/MIMIC-train"

ACCOUNT_ARGS=()
if [[ -n "${ACCOUNT}" ]]; then
  ACCOUNT_ARGS=(--account="${ACCOUNT}")
fi

# Same BLAS/thread-pool pinning rationale as clean_LOS/run_all.slurm: a
# numpy/sklearn op inside a multi-CPU job can otherwise spawn far more
# threads than allocated on a shared node, causing contention that surfaces
# as native heap corruption / SIGSEGV rather than a clean Python error.
ENV_SETUP="module load container_env python3/2024.2-py310; export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-1} OPENBLAS_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-1} NUMEXPR_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-1} VECLIB_MAXIMUM_THREADS=\${SLURM_CPUS_PER_TASK:-1};"
PYRUN_DATAPREP="crun -p ${DATAPREP_ENV} python"
PYRUN_TRAIN="crun -p ${TRAIN_ENV} python"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}/output" "${LOG_DIR}/error"

# Project root: MIMIC_III -> BIBM_versions -> sibtanu-time-series-research ->
# project root (3 levels up) -- matches every Python script's own
# _PROJECT_ROOT computation. Only used here for the human-readable echo at
# the end and the combine_results.py --results_dir arg; every script below
# already self-locates everything else via __file__.
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RESULTS_DIR="${PROJECT_ROOT}/results_timeseries"

# 6 architectures: (job_tag, subdirectory, script filename) triples.
declare -a JOB_TAGS=(bilstm hybrid ome patchtst puremamba transformer)
declare -a SUBDIRS=(
  "BiLSTM_with_static_ANN_fusion"
  "Hybrid_BiLSTM_BiMAMBA_Static_ANN_Fusion"
  "OME_BiLSTM_Transformer_Mamba_Static_ANN"
  "PatchTST_style_Transformer_Static_ANN_Fusion"
  "Pure_Mamba_Static_ANN_Fusion"
  "Transformer_Static_ANN_Fusion"
)
declare -a SCRIPTS=(
  "V1_train_mimic3_lstm_fusion_multiclass_auc.py"
  "V1_train_mimic3_hybrid_lstm_mamba_multiclass_auc.py"
  "V1_train_mimic3_ome_fusion_multiclass_auc.py"
  "V1_train_mimic3_patchtst_fusion_multiclass_auc.py"
  "V1_train_mimic3_pure_mamba_multiclass_auc.py"
  "V1_train_mimic3_transformer_fusion_multiclass_auc.py"
)
# Per-model CLI overrides on top of the standard hyperparameters -- empty for
# 5 of 6 (true apples-to-apples comparison). Pure-Mamba is the disclosed
# exception: under the shared lr=1e-3 it produced NaN/Inf test probabilities
# and the run was refused (the script's own safety check, not a crash) --
# confirmed via a manual retry that --lr 3e-4 --grad_clip 0.5 --layers 2
# converges. Worth reporting as a real finding about this architecture's
# robustness, not silently normalized away.
declare -a EXTRA_ARGS=(
  ""
  ""
  ""
  ""
  "--lr 3e-4 --grad_clip 0.5 --layers 2"
  ""
)

# ------------------------------------------------------------------------------
# Stage 0: data pipeline (build -> impute -> preprocess), CPU partition.
# Reads raw CSVs from original_data/, writes intermediate wide CSVs + final
# model-ready tensors under processed_data/structured_LOS_dynamic/.
# ------------------------------------------------------------------------------
submit_build_job () {
  sbatch --parsable \
         "${ACCOUNT_ARGS[@]}" \
         --partition="${PARTITION_CPU}" \
         --job-name="ts_build" \
         --output="${LOG_DIR}/output/ts_build_%j.log" \
         --error="${LOG_DIR}/error/ts_build_%j.log" \
         --cpus-per-task=4 \
         --mem=64G \
         --time=08:00:00 \
         --wrap="${ENV_SETUP} ${PYRUN_DATAPREP} '${SCRIPT_DIR}/V12.1_build_script.py' --cohort_filter ${COHORT_FILTER}"
}

submit_impute_job () {
  local dep=$1
  sbatch --parsable \
         "${ACCOUNT_ARGS[@]}" \
         --partition="${PARTITION_CPU}" \
         --dependency=afterok:${dep} \
         --job-name="ts_impute" \
         --output="${LOG_DIR}/output/ts_impute_%j.log" \
         --error="${LOG_DIR}/error/ts_impute_%j.log" \
         --cpus-per-task=8 \
         --mem=32G \
         --time=04:00:00 \
         --wrap="${ENV_SETUP} ${PYRUN_DATAPREP} '${SCRIPT_DIR}/V12.1_impute_script.py' --imputer ${IMPUTER}"
}

submit_preprocess_job () {
  local dep=$1
  sbatch --parsable \
         "${ACCOUNT_ARGS[@]}" \
         --partition="${PARTITION_CPU}" \
         --dependency=afterok:${dep} \
         --job-name="ts_preprocess" \
         --output="${LOG_DIR}/output/ts_preprocess_%j.log" \
         --error="${LOG_DIR}/error/ts_preprocess_%j.log" \
         --cpus-per-task=4 \
         --mem=32G \
         --time=01:00:00 \
         --wrap="${ENV_SETUP} ${PYRUN_DATAPREP} '${SCRIPT_DIR}/V12.1_preprocess_script.py'"
}

echo "[INFO] Submitting Stage 0 (data pipeline): build -> impute -> preprocess..."
BUILD_JOB=$(submit_build_job)
IMPUTE_JOB=$(submit_impute_job "${BUILD_JOB}")
PREPROCESS_JOB=$(submit_preprocess_job "${IMPUTE_JOB}")
echo "  build=${BUILD_JOB} -> impute=${IMPUTE_JOB} -> preprocess=${PREPROCESS_JOB}"

# ------------------------------------------------------------------------------
# Stage 1: train all 6 multiclass (plain 3-class, non-ordinal) architectures,
# GPU partition. Each depends only on preprocess finishing (afterok) -- the
# 6 are independent of each other and run in parallel once preprocess is done.
# ------------------------------------------------------------------------------
submit_train_job () {
  local job_tag=$1 script_path=$2 dep=$3 extra_args=$4
  local job_name="ts_${job_tag}"

  sbatch --parsable \
         "${ACCOUNT_ARGS[@]}" \
         --partition="${PARTITION_GPU}" \
         --gres=gpu:"${GPUS_PER_JOB}" \
         --dependency=afterok:${dep} \
         --job-name="${job_name}" \
         --output="${LOG_DIR}/output/${job_name}_%j.log" \
         --error="${LOG_DIR}/error/${job_name}_%j.log" \
         --cpus-per-task=4 \
         --mem=32G \
         --time=04:00:00 \
         --wrap="${ENV_SETUP} ${PYRUN_TRAIN} '${script_path}' --workers 4 ${extra_args}"
}

echo "[INFO] Submitting Stage 1 (6 multiclass training jobs)..."
ALL_TRAIN_JOB_IDS=()

for i in "${!JOB_TAGS[@]}"; do
  tag="${JOB_TAGS[$i]}"
  script_path="${SCRIPT_DIR}/Training/Multiclass/${SUBDIRS[$i]}/${SCRIPTS[$i]}"

  if [[ ! -f "${script_path}" ]]; then
    echo "[ERROR] Missing script: ${script_path}" >&2
    exit 1
  fi

  job_id=$(submit_train_job "${tag}" "${script_path}" "${PREPROCESS_JOB}" "${EXTRA_ARGS[$i]}")
  echo "  submitted ${tag} -> job ${job_id}"
  ALL_TRAIN_JOB_IDS+=("${job_id}")
done

# ------------------------------------------------------------------------------
# Stage 2: combine all 6 models' test metrics into one CSV (micro + weighted
# scores) once every training job has finished -- afterany (not afterok) so
# a partial sweep (e.g. one model crashed) still produces a summary of
# whichever models did succeed. combine_results.py is stdlib-only (json/csv/
# glob/argparse), so it runs fine in DATAPREP_ENV -- no need for the GPU
# partition or the torch-heavy TRAIN_ENV just to write a CSV.
# ------------------------------------------------------------------------------
DEP=$(IFS=:; echo "${ALL_TRAIN_JOB_IDS[*]}")
COMBINE_JOB=$(sbatch --parsable \
       "${ACCOUNT_ARGS[@]}" \
       --partition="${PARTITION_CPU}" \
       --dependency=afterany:${DEP} \
       --job-name="ts_combine_results" \
       --output="${LOG_DIR}/output/ts_combine_results_%j.log" \
       --error="${LOG_DIR}/error/ts_combine_results_%j.log" \
       --cpus-per-task=1 \
       --mem=4G \
       --time=00:10:00 \
       --wrap="${ENV_SETUP} ${PYRUN_DATAPREP} '${SCRIPT_DIR}/Training/Multiclass/combine_results.py' --results_dir '${RESULTS_DIR}'")

echo "[INFO] Submitted: build=${BUILD_JOB} impute=${IMPUTE_JOB} preprocess=${PREPROCESS_JOB}"
echo "[INFO] Submitted ${#ALL_TRAIN_JOB_IDS[@]} training jobs: ${ALL_TRAIN_JOB_IDS[*]}"
echo "[INFO] Submitted combine job: ${COMBINE_JOB}"
echo "Final summary will be written to ${RESULTS_DIR}/multiclass_summary.csv"
