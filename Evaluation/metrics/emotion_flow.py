import numpy

from sklearn.metrics import f1_score, precision_score, accuracy_score

from . import Task
from ..utils.io import load_json, load_pickle
from ..utils.expand import expand_indexed_batches, expand_round_time


class EmotionFlow(Task):
    @classmethod
    def pre_process(cls, goldens_filepath, results_filepath, alltime, alltime_type) -> tuple[list[int], list[int], list[list[float]]]:
        goldens = load_json(goldens_filepath)
        goldens, goldens_batch_sizes = expand_indexed_batches(goldens)

        results = load_json(results_filepath)
        results, results_batch_sizes = expand_indexed_batches(results)

        assert len(goldens_batch_sizes) == len(results_batch_sizes)
        assert sum(goldens_batch_sizes) == sum(results_batch_sizes)

        multiple_inference_times: list[list[float]] = list()
        for round_time in alltime[alltime_type]:
            round_time = expand_round_time(round_time, results_batch_sizes)
            multiple_inference_times.append(round_time)

        return goldens, results, multiple_inference_times


    @classmethod
    def calculate_metrics(cls, goldens: list[int], results: list[int], validities: numpy.ndarray) -> tuple[float, float]:
        # each element of inference_results must be a list
        penalty = float(sum(validities)/len(validities))
        if len(results) == 0:
            return float('NaN'), penalty

        invalid_result = min(goldens) - 1

        masked_results = list()
        for result, validity in zip(results, validities):
            result = result if validity else invalid_result
            masked_results.append(result)
        score = f1_score(goldens, masked_results, average='weighted')
        return score, penalty
        # return f1_score(goldens, masked_results, average='weighted')
        # return accuracy_score(goldens, masked_results)
        #return numpy.where(validities == False)[0], precision_score(goldens, masked_results, average='weighted')