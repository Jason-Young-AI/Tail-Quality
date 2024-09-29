import tqdm
import numpy
import itertools
import multiprocessing

from typing import Any, Callable


def tail_quality(quality_calculation_function: Callable[[tuple[Any, Any, list[numpy.ndarray]]], Any], inference_goldens: Any, inference_results: Any, multiple_inference_times: list[list[float]], thresholds: list[float], worker_number: int, worker_batch_size: int) -> list[list[Any]]:
    # the length (s) of each element of `multiple_inference_times` must be equal to the length (s) of `inference_results`.
    #
    # the length of `inference_results` is (s).
    #
    # the length of `multiple_inference_times` is (n).
    #
    # the length of thresholds is (m).
    #
    # return: q - quality; t - threshold
    #        [ [q_1 @ t_1, q_2 @ t_1, ...,  q_n @ t_1],  [q_1 @ t_2, q_2 @ t_2, ...,  q_n @ t_2], ..., [q_1 @ t_m, q_2 @ t_m, ...,  q_n @ t_m]]

    all_qualities_at_thresholds = list()
    all_penalty_at_thresholds = list()

    inference_validities_at_thresholds: list[list[numpy.ndarray]] = list()
    # [
    #   [
    #     numpy.array( ( len(s) ), dtype=numpy.bool),
    #     ... 
    #   ] (length = n),
    #   ...
    # ] (length = m)
    print(f'Checking Validity')
    for threshold in thresholds:
        inference_validities_at_threshold: list[numpy.ndarray] = list()
        for inference_times in multiple_inference_times:
            inference_validities_at_threshold.append(numpy.array([inference_time <= threshold for inference_time in inference_times], dtype=bool))
        inference_validities_at_thresholds.append(inference_validities_at_threshold)
    print(f'Done')

    # sorted_qts = list()
    for inference_validities_at_threshold, threshold in zip(inference_validities_at_thresholds, thresholds):
        parameters = [(inference_goldens, inference_results, inference_validities_at_threshold[start_index:start_index+worker_batch_size]) for start_index in range(0, len(inference_validities_at_threshold), worker_batch_size)]
        # parameters = [(inference_goldens, inference_results, inference_validities) for inference_validities in inference_validities_at_threshold]
        all_qualities_at_threshold = list()
        all_penalty_at_threshold = list()
        with multiprocessing.Pool(worker_number) as pool:
            with tqdm.tqdm(total=len(inference_validities_at_threshold), desc=f'Calculating Quality @ Threshold={threshold}') as progress_bar:
                # for index, (vali, quality_at_threshold) in enumerate(pool.imap_unordered(quality_calculation_function, parameters), start=1):
                for index, (batch_qualities_list, batch_penalty_list) in enumerate(pool.imap_unordered(quality_calculation_function, parameters), start=1):
                    all_qualities_at_threshold.extend(batch_qualities_list)
                    all_penalty_at_threshold.extend(batch_penalty_list)
                    progress_bar.update(len(batch_penalty_list))
        # sorted_qts.append(sorted(qualities_at_threshold, key=lambda x: x[1]))
        all_qualities_at_thresholds.append(all_qualities_at_threshold)
        all_penalty_at_thresholds.append(all_penalty_at_threshold)
    # print(sorted_qts[-2][0])
    # print(sorted_qts[-2][-1])

    # print(sorted_qts[-1][0])
    # print(sorted_qts[-1][-1])

    return all_qualities_at_thresholds, all_penalty_at_thresholds
