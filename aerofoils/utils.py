from time import perf_counter as now


class Timer:

    def __enter__(self):
        self.start = now()
        return self

    def __exit__(self, type, value, traceback):
        self.stop = now()
        self.elapsed = self.stop - self.start
