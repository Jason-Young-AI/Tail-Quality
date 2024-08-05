import numpy

from typing import Any

class Task(object):
    @classmethod
    def pre_process(cls, goldens_filepath, results_filepath, alltime_filepath) -> tuple[Any, Any, list[list[float]]]:
        # This classmethod should return goldens, results, multiple_inference_times
        raise NotImplementedError

    @classmethod
    def get_metrics(cls, parameters: tuple[ Any, Any, numpy.ndarray ]) -> Any:
        goldens, results, validities = parameters
        qualities = cls.calculate_metrics(goldens, results, validities)
        return qualities

    @classmethod
    def calculate_metrics(cls, goldens: Any, results: Any, validities: numpy.ndarray) -> Any:
        raise NotImplementedError