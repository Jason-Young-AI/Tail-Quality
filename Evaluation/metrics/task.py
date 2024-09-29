import numpy

from typing import Any

class Task(object):
    @classmethod
    def pre_process(cls, goldens_filepath, results_filepath, alltime_filepath) -> tuple[Any, Any, list[list[float]]]:
        # This classmethod should return goldens, results, multiple_inference_times
        raise NotImplementedError

    @classmethod
    def get_metrics(cls, parameters: tuple[ Any, Any, numpy.ndarray ]) -> tuple[list[Any], list[float]]:
        goldens, results, validities_list = parameters
        qualities_list = list()
        penalty_list = list()
        for validities in validities_list:
            qualities, penalty = cls.calculate_metrics(goldens, results, validities)
            qualities_list.append(qualities)
            penalty_list.append(penalty)
        return (qualities_list, penalty_list)

    @classmethod
    def calculate_metrics(cls, goldens: Any, results: Any, validities: numpy.ndarray) -> tuple[Any, float]:
        raise NotImplementedError