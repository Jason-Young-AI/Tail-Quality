import numpy

from . import Task
from ..utils.io import load_pickle
from ..utils.expand import expand_indexed_batches, expand_round_time, get_sorted_batch


def get_triple_metrics(goldens, results):
    topk = 20

    predictions = list()
    max_predictions = list()
    for golden, result in zip(goldens, results):
        # golden: [golden_1, golden_2, ..., golden_n]
        # result: [result_1, result_2, ..., result_m]
        # validity: bool
        predictions.append(numpy.array([(recommend in golden) for recommend in result], dtype=bool))
        max_predictions.append(numpy.array([1 if i < len(golden) else 0 for i in range(topk)]))
    predictions = numpy.array(predictions)
    max_predictions = numpy.array(max_predictions)

    tp = numpy.sum(predictions[:, :topk], axis=1)
    precision_n = topk
    recall_n = numpy.array([len(golden) for golden in goldens])
    precision = numpy.sum(tp) / topk
    recall = numpy.sum(tp / recall_n)

    idcg = numpy.sum(max_predictions * 1.0 / numpy.log2(numpy.arange(2, topk + 2)), axis=1)
    dcg = predictions * (1.0 / numpy.log2(numpy.arange(2, topk + 2)))
    dcg = numpy.sum(dcg, axis=1)
    idcg[idcg == 0.0] = 1.0
    ndcg = dcg / idcg
    ndcg[numpy.isnan(ndcg)] = 0.0
    ndcg = numpy.sum(ndcg)
    return precision, recall, ndcg


class LightGCNFaster(Task):
    @classmethod
    def pre_process(cls, goldens_filepath, results_filepath, alltime_filepath, alltime_type) -> tuple[list[int], list[int], list[list[float]]]:
        goldens = load_pickle(goldens_filepath)
        goldens = [golden for index, golden in get_sorted_batch(goldens)]

        results = load_pickle(results_filepath)
        results = [result for index, result in get_sorted_batch(results)]
        triple_results = list()
        for golden, result in zip(goldens, results):
            precision, recall, ndcg = get_triple_metrics(golden, result)
            triple_results.append((precision, recall, ndcg))

        alltime = load_pickle(alltime_filepath)[alltime_type]
        multiple_inference_times: list[list[float]] = list()
        for round_time in alltime:
            round_time = [batch_time for index, batch_time in get_sorted_batch(round_time)]
            multiple_inference_times.append(round_time)

        return goldens, triple_results, multiple_inference_times


    @classmethod
    def calculate_metrics(cls, goldens: list[list[int]], results: list[list[int]], validities: numpy.ndarray) -> tuple[float, float, float]:
        # Return Precision, Recall, Normalized Discounted Cumulative Gain
        # each element of inference_results must be a list
        if len(results) == 0:
            return float('NaN')
        total_precision = 0
        total_recall = 0
        total_ndcg = 0
        total = 0
        for golden, (precision, recall, ndcg), validity in zip(goldens, results, validities):
            total_precision += precision if validity else 0
            total_recall += recall if validity else 0
            total_ndcg += ndcg if validity else 0
            total += len(golden)

        precision = total_precision / total
        recall = total_recall / total
        ndcg = total_ndcg / total
        return precision, recall, ndcg
