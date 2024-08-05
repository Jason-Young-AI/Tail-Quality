import io
import sys
import numpy

# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval
from faster_coco_eval import COCO, COCOeval_faster

from . import Task
from ..utils.io import load_json, load_pickle
from ..utils.expand import expand_indexed_batches, expand_round_time


def coco2numpy(coco_results):
    numpy_results = numpy.ndarray((len(coco_results), 7))

    for index, coco_result in enumerate(coco_results):
        numpy_results[index][0] = coco_result['image_id']
        numpy_results[index][1] = coco_result['bbox'][0]
        numpy_results[index][2] = coco_result['bbox'][1]
        numpy_results[index][3] = coco_result['bbox'][2]
        numpy_results[index][4] = coco_result['bbox'][3]
        numpy_results[index][5] = coco_result['score']
        numpy_results[index][6] = coco_result['category_id']

    return numpy_results


class DETR(Task):
    @classmethod
    def pre_process(cls, goldens_filepath, results_filepath, alltime_filepath, alltime_type) -> tuple[COCO, numpy.ndarray, list[list[float]]]:
        goldens = COCO(str(goldens_filepath))

        results = load_json(results_filepath)

        results, batch_sizes = expand_indexed_batches(results)

        alltime = load_pickle(alltime_filepath)[alltime_type]
        multiple_inference_times: list[list[float]] = list()
        for round_time in alltime:
            round_time = expand_round_time(round_time, batch_sizes)
            multiple_inference_times.append(round_time)

        detail_sizes = [len(result) for result in results]
        detail_results = [coco2numpy(result) for result in results]
        detail_multiple_inference_times: list[list[float]] = list()
        for round_time in multiple_inference_times:
            new_round_time = list()
            for detail_time, detail_size in zip(round_time, detail_sizes):
                new_round_time.extend([detail_time for _ in range(detail_size)])
            detail_multiple_inference_times.append(new_round_time)

        return goldens, numpy.concatenate(detail_results, axis=0), detail_multiple_inference_times


    @classmethod
    def calculate_metrics(cls, goldens: COCO, results: numpy.ndarray, validities: numpy.ndarray) -> list[float]:
        if len(results) == 0:
            return float('NaN')

        assert len(results) == len(validities)
        #masked_results = list()
        #for result, validity in zip(results, validities):
        #    if validity:
        #        masked_results.append(result)
        masked_results = results[validities == True]

        if len(masked_results) == 0:
            return float('NaN')

        #original_stdout = sys.stdout
        #sys.stdout = io.StringIO()

        #masked_results = numpy.concatenate(masked_results, axis=0)
        results = goldens.loadRes(masked_results)
        coco_eval = COCOeval_faster(goldens, results, iouType='bbox')
        coco_eval.params.iouThrs = numpy.array([0.5])
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        #sys.stdout = original_stdout
        return coco_eval.stats.tolist()
