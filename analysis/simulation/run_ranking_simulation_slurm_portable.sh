#!/bin/bash
# Portable Slurm launcher for the locked v77 1,620-cell array.
# Set REPRO_ROOT and PYTHON before submitting, for example:
#   REPRO_ROOT=$PWD PYTHON=$(command -v python) sbatch --export=ALL analysis/simulation/run_ranking_simulation_slurm_portable.sh
#SBATCH --job-name=haplo-rank-v77
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-1619%36

set -euo pipefail

: "${REPRO_ROOT:?Set REPRO_ROOT to the repository root}"
: "${PYTHON:?Set PYTHON to the production-environment interpreter}"

CODE="$REPRO_ROOT/analysis/simulation"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPRO_ROOT/work/ranking-simulation-v77}"
LEAD_FREQUENCY_CLASSES=(rare common)
SCENARIOS=(lead_ld_q1 balanced_q1 lead_ld_q2)
SAMPLES=(500 1000 2000)
PARTNERS=(4 32 256)

frequency_block=$((3 * 3 * 3 * 30))
scenario_block=$((3 * 3 * 30))
sample_block=$((3 * 30))
partner_block=30
frequency_index=$((SLURM_ARRAY_TASK_ID / frequency_block))
remainder=$((SLURM_ARRAY_TASK_ID % frequency_block))
scenario_index=$((remainder / scenario_block))
remainder=$((remainder % scenario_block))
sample_index=$((remainder / sample_block))
remainder=$((remainder % sample_block))
partner_index=$((remainder / partner_block))
replicate=$((remainder % partner_block))

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONPATH="$REPRO_ROOT/analysis/estimation:$CODE${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" -u "$CODE/run_ranking_simulation_cell_v77.py" \
  --lead-frequency-class "${LEAD_FREQUENCY_CLASSES[$frequency_index]}" \
  --scenario "${SCENARIOS[$scenario_index]}" \
  --n-haplotypes "${SAMPLES[$sample_index]}" \
  --partners "${PARTNERS[$partner_index]}" \
  --replicate-start "$replicate" \
  --replicates 1 \
  --output-root "$OUTPUT_ROOT"
