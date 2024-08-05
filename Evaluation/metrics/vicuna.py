import numpy

from . import Task
from ..utils.io import load_json, load_pickle
from ..utils.expand import expand_indexed_batches, expand_round_time


class Vicuna(Task):
    @classmethod
    def pre_process(cls, goldens_filepath, results_filepath, alltime_filepath, alltime_type) -> tuple[list[str], list[str], list[list[float]]]:
        goldens = load_json(goldens_filepath)
        goldens, goldens_batch_sizes = expand_indexed_batches(goldens)

        results = load_json(results_filepath)
        results, results_batch_sizes = expand_indexed_batches(results)

        assert len(goldens_batch_sizes) == len(results_batch_sizes)
        assert sum(goldens_batch_sizes) == sum(results_batch_sizes)

        alltime = load_pickle(alltime_filepath)[alltime_type]
        multiple_inference_times: list[list[float]] = list()
        for round_time in alltime:
            round_time = expand_round_time(round_time, results_batch_sizes)
            multiple_inference_times.append(round_time)

        return goldens, results, multiple_inference_times


    @classmethod
    def calculate_metrics(cls, goldens: list[str], results: list[str], validities: numpy.ndarray) -> float:
        # each element of inference_results must be a list
        if len(results) == 0:
            return float('NaN')

        right = 0
        total = 0
        for golden, result, validity in zip(goldens, results, validities):
            right += (validity and (golden in result))
            total += 1
        accuracy = right / total
        return accuracy