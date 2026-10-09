"""Parallel map over a cohort's cells."""

import os
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

N_WORKERS = max(1, (os.cpu_count() or 1) - 1)


def _limit_blas_threads() -> None:
    # Workers are already one-per-core; letting each spawn a BLAS thread pool
    # oversubscribes and runs slower than serial.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"


def run_jobs(cells, job, *job_args) -> list[tuple]:
    """Map `job` over a list of cells, in parallel when there is more than one core.

    Takes the cells rather than a cohort, since nothing here reads a label.

    Cells vary ~100x in node count (median 2.7k, max 32k), so chunksize=1 --
    otherwise the workers holding the largest cells become stragglers.
    ProcessPoolExecutor.map preserves input order, keeping row order reproducible.
    """
    if N_WORKERS <= 1 or len(cells) < 2:
        return [job(cell, *job_args) for cell in cells]
    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_limit_blas_threads) as ex:
        args = [repeat(a, len(cells)) for a in job_args]
        return list(ex.map(job, cells, *args, chunksize=1))
