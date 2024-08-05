import numpy

from . import Task
from ..utils.io import load_pickle
from ..utils.expand import expand_indexed_batches, expand_round_time


class MobileNet(Task):
    @classmethod
    def pre_process(cls, goldens_filepath, results_filepath, alltime_filepath, alltime_type) -> tuple[list[str], list[str], list[list[float]]]:
        goldens = load_pickle(goldens_filepath)
        goldens, goldens_batch_sizes = expand_indexed_batches(goldens)

        results = load_pickle(results_filepath)
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
    def calculate_metrics(cls, goldens: list[int], results: list[int], validities: numpy.ndarray) -> tuple[float, float]:
        # each element of inference_results must be a list
        if len(results) == 0:
            return float('NaN')

        top1_right = 0
        top5_right = 0
        total = 0
        for golden, (top1_result, top5_result), validity in zip(goldens, results, validities):
            top1_right += (validity and (golden in top1_result))
            top5_right += (validity and (golden in top5_result))
            total += 1
        top1_accuracy = top1_right / total
        top5_accuracy = top5_right / total
        return (top1_accuracy, top5_accuracy)