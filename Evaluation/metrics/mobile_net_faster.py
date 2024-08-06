import numpy

from . import Task
from ..utils.io import load_pickle
from ..utils.expand import expand_indexed_batches, expand_round_time, get_sorted_batch


class MobileNetFaster(Task):
    @classmethod
    def pre_process(cls, goldens_filepath, results_filepath, alltime_filepath, alltime_type) -> tuple[None, list[tuple[float, float]], list[list[float]]]:
        goldens = load_pickle(goldens_filepath)
        goldens = [golden for index, golden in get_sorted_batch(goldens)]

        results = load_pickle(results_filepath)
        results = [result for index, result in get_sorted_batch(results)]

        top_results = list()
        for golden, result in zip(goldens, results):
            top1_right = 0
            top5_right = 0
            for golden_item, (top1_result, top5_result) in zip(golden, result):
                top1_right += int(golden_item in top1_result)
                top5_right += int(golden_item in top5_result)
            top_results.append((top1_right/len(golden), top5_right/len(golden)))

        alltime = load_pickle(alltime_filepath)[alltime_type]
        multiple_inference_times: list[list[float]] = list()
        for round_time in alltime:
            round_time = [batch_time for index, batch_time in get_sorted_batch(round_time)]
            multiple_inference_times.append(round_time)

        return None, top_results, multiple_inference_times

    @classmethod
    def calculate_metrics(cls, goldens: None, results: list[tuple[float, float]], validities: numpy.ndarray) -> tuple[float, float]:
        # each element of inference_results must be a list
        if len(results) == 0:
            return float('NaN')

        top1_acc = 0
        top5_acc = 0
        total = 0
        for (top1_result, top5_result), validity in zip(results, validities):
            top1_acc += top1_result if validity else 0
            top5_acc += top5_result if validity else 0
            total += 1
        top1_accuracy = top1_acc / total
        top5_accuracy = top5_acc / total
        return (top1_accuracy, top5_accuracy)
