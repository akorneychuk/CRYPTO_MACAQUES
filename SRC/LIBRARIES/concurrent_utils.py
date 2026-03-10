import time

from SRC.CORE.debug_utils import produce_measure, is_running_under_pycharm
from concurrent.futures import ThreadPoolExecutor, as_completed


def iterate_multithread_executor_safe(executor, _func, args, num_workers=1):
    if num_workers > 1:
        futures = {executor.submit(_func, i): i for i in args}
        try:
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=30)

                    yield result
                except Exception as e:
                    yield f"Error processing {futures[future]}: {e}"  # Handle exceptions
        finally:
            for f in futures:
                if not f.done():
                    f.cancel()

            for f in futures:
                try:
                    f.result(timeout=10)
                except:
                    pass
    else:
        for arg in args:
            yield _func(arg)


def iterate_multithread_executor(_func, args, num_workers=1, thread_name_prefix='MULTITHREAD_BG'):
    with ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix=thread_name_prefix) as executor:
        try:
            yield from iterate_multithread_executor_safe(executor, _func, args, num_workers=num_workers)
        finally:
            executor.shutdown(wait=False)


def iterate_multiprocess_executor(_func, args, num_workers=1):
    if num_workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        try:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(_func, i) for i in args]

                for future in as_completed(futures):
                    result = future.result()
                    yield result
        finally:
            executor.shutdown()
    else:
        for arg in args:
            yield _func(arg)


def iterate_multiprocess_pool_safe(_func, args, num_workers=1):
    if num_workers <= 1 or is_running_under_pycharm():
        for arg in args:
            yield _func(arg)
        return

    from multiprocessing import Pool
    pool = Pool(num_workers)

    try:
        for result in pool.imap_unordered(_func, args):
            yield result

        # normal finish
        pool.close()
        pool.join()

    except GeneratorExit:
        # caller stopped iteration
        pool.terminate()
        pool.join()
        raise

    except Exception:
        pool.terminate()
        pool.join()
        raise


def iterate_multiprocess_pool(_func, args, num_workers=1):
    if num_workers > 1:
        from multiprocessing import Pool
        try:
            with Pool(num_workers) as pool:
                for result in pool.imap_unordered(_func, args):
                    yield result
        finally:
            pool.terminate()
            pool.join()
    else:
        for arg in args:
            yield _func(arg)


class SharedEvent:
    def __init__(self, manager):
        self._flag = manager.Value('b', False)

    def set(self):
        self._flag.value = True

    def clear(self):
        self._flag.value = False

    def is_set(self):
        return self._flag.value


def test_job_func(arg):
    time.sleep(arg)

    return arg


if __name__ == "__main__":
    args = [15, 17, 10, 12, 20]
    num_workers = 5

    measure = produce_measure()
    for result in iterate_multiprocess_executor(test_job_func, args, num_workers=num_workers):
        print(f"EXECUTOR Task {result} is complete")
    print(f"EXECUTOR COMPLETED: {measure()}")

    measure = produce_measure()
    for result in iterate_multiprocess_pool(test_job_func, args, num_workers=num_workers):
        print(f"POOL Task {result} is complete")
    print(f"POOL COMPLETED: {measure()}")