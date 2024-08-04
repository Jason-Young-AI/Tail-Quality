import io
import sys
import numpy

from typing import Any

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def calculate_acc(parameters: tuple[list[list[Any]], list[Any], numpy.ndarray]) -> float:
    # each element of inference_results must be a list
    inference_goldens, inference_results, inference_validities = parameters
    if len(inference_results) == 0:
        return float('NaN')

    right = 0
    total = 0
    for golden, result, validity_at_threshold in zip(inference_goldens, inference_results, inference_validities):
        right += (validity_at_threshold and (golden in result))
        total += 1
    accuracy = right / total
    return accuracy


def calculate_map(parameters: tuple[COCO, numpy.ndarray, numpy.ndarray]) -> numpy.ndarray:
    inference_goldens, inference_results, inference_validities = parameters
    if len(inference_results) == 0:
        return float('NaN')

    inference_results = inference_results[inference_validities]

    inference_results = inference_goldens.loadRes(inference_results)
    coco_eval = COCOeval(inference_goldens, inference_results, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats