'''
STAMP: shared per-item multiprocessing helper
'''

# Import external dependencies
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from typing import TypeVar

# Import internal STAMP objects
from stamp.utils.log import get_worker_log_config, init_worker_logging, log

T = TypeVar('T')
R = TypeVar('R')

# run_parallel: run worker_fn(item) for each item, sequentially in-process if max_workers <= 1, else pooled
def run_parallel(
    items: list[T],
    worker_fn: Callable[[T], R],
    *,
    max_workers: int = 1,
    label: str,
    on_success: Callable[[T, R], None] | None = None,
    on_error: Callable[[T, Exception], None] | None = None,
    pool: ProcessPoolExecutor | None = None,
    progress_total: int | None = None,
    progress_offset: int = 0,
) -> list[R]:
    total = len(items)
    if total == 0:
        return []
    # progress_total lets callers define an overall total to report against (eg if using batches)
    report_total = progress_total if progress_total is not None else total
    report_every = max(1, report_total // 10)
    results: list[R] = []

    # _handle: per-item bookkeeping
    def _handle(item: T, index: int, get_result: Callable[[], R]) -> None:
        try:
            result = get_result()
        except Exception as exc:
            if on_error is None:
                raise
            on_error(item, exc)
        else:
            if on_success is not None:
                on_success(item, result)
            results.append(result)
        reported_index = progress_offset + index
        if reported_index % report_every == 0 or reported_index == report_total:
            log.progress(f'{label}: {reported_index}/{report_total} complete')

    if max_workers <= 1 or total == 1:
        for index, item in enumerate(items, start=1):
            _handle(item, index, lambda item=item: worker_fn(item))
        return results

    # use the provided pool when given
    if pool is not None:
        future_to_item = {pool.submit(worker_fn, item): item for item in items}
        for index, future in enumerate(as_completed(future_to_item), start=1):
            item = future_to_item[future]
            _handle(item, index, future.result)
        return results

    level_name, log_path = get_worker_log_config()
    with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker_logging, initargs=(level_name, log_path)) as owned_pool:
        future_to_item = {owned_pool.submit(worker_fn, item): item for item in items}
        for index, future in enumerate(as_completed(future_to_item), start=1):
            item = future_to_item[future]
            _handle(item, index, future.result)
    return results

# _call_indexed: wrapper pairing a worker's result with its original index
def _call_indexed(worker_fn: Callable[[T], R], indexed: tuple[int, T]) -> tuple[int, R]:
    index, item = indexed
    return index, worker_fn(item)

# run_parallel_ordered: call run_parallel but return results in input order (for order-sensitive call sites, e.g. feature matrices)
def run_parallel_ordered(
    items: list[T],
    worker_fn: Callable[[T], R],
    *,
    max_workers: int = 1,
    label: str,
    pool: ProcessPoolExecutor | None = None,
    progress_total: int | None = None,
    progress_offset: int = 0,
) -> list[R]:
    indexed_results = run_parallel(
        list(enumerate(items)),
        partial(_call_indexed, worker_fn),
        max_workers=max_workers,
        label=label,
        pool=pool,
        progress_total=progress_total,
        progress_offset=progress_offset,
    )
    return [result for _, result in sorted(indexed_results, key=lambda pair: pair[0])]
