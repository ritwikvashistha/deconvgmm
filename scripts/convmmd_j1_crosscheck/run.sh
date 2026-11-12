#!/bin/bash
# Launch cross-check workers, one per core, on a queue built by build_reference.py.
#
# usage: scripts/convmmd_j1_crosscheck/run.sh <work_dir> <first_core> <n_workers>
#
# Each worker is pinned to one core (Linux taskset) with single-threaded XLA and
# takes units from <work_dir>/queue.json until it is empty; results are appended
# to <work_dir>/results.jsonl, and <work_dir>/done.txt is written when all exit.
# PY selects the interpreter (default: python); put JAX on PYTHONPATH if needed.
# Run under nohup for long jobs: the full 780-unit queue took about 8.5 h on 16
# cores of a shared server.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
work=$(cd "$1" && pwd); first=$2; workers=$3
PY=${PY:-python}
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
       XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
for i in $(seq 0 $((workers - 1))); do
  nohup taskset -c $((first + i)) "$PY" -u "$here/crosscheck.py" --ref "$work/j1_ref.json" \
        --out "$work/results.jsonl" --queue "$work/queue.json" --counter "$work/counter.txt" \
        > "$work/worker_$i.log" 2>&1 &
done
wait
echo ALLDONE >> "$work/done.txt"
