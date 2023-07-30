import io
import sys
import numpy
import scipy
import pathlib
import argparse

from extract_data import extract_data
from constant import dataset_choices, combine_choices, quality_choices


def calculate_stat(cts):
    if len(cts):
        kurts = scipy.stats.kurtosis(cts, axis=-1, bias=False, fisher=True)
        skews = scipy.stats.skew(cts, axis=-1, bias=False)
        mins = numpy.min(cts, axis=-1)
        maxs = numpy.max(cts, axis=-1)
        meds = numpy.median(cts, axis=-1)
        avgs = numpy.average(cts, axis=-1)
        vars = numpy.var(cts, axis=-1, ddof=1)
        stds = numpy.std(cts, axis=-1, ddof=1)
        q1s = numpy.quantile(cts, 0.25, axis=-1)
        q3s = numpy.quantile(cts, 0.75, axis=-1)
        iqrs = q3s - q1s
        lower_whiskers = q1s - 1.5 * iqrs
        upper_whiskers = q3s + 1.5 * iqrs
    else:
        kurts = numpy.array([])
        skews = numpy.array([])
        mins = numpy.array([])
        maxs = numpy.array([])
        meds = numpy.array([])
        avgs = numpy.array([])
        vars = numpy.array([])
        stds = numpy.array([])
        q1s = numpy.array([])
        q3s = numpy.array([])
        iqrs = numpy.array([])
        lower_whiskers = numpy.array([])
        upper_whiskers = numpy.array([])

    return dict(
        kurts = kurts,
        skews = skews,
        mins = mins,
        maxs = maxs,
        meds = meds,
        avgs = avgs,
        vars = vars,
        stds = stds,
        q1s = q1s,
        q3s = q3s,
        iqrs = iqrs,
        lower_whiskers = lower_whiskers,
        upper_whiskers = upper_whiskers
    )


def combine_times(times, combine_type):
    # Arugment 'times' should contain 3 keys: inference_time, preprocess_time, postprocess_time
    # Each value of the corresponding key must be an numpy.array object with shape [instance_number, ]
    assert combine_type in combine_choices, f"Wrong Type of Combine Type: {combine_type}"

    inf_time = times['inference_times']
    pre_time = times['preprocess_times']
    post_time = times['postprocess_times']

    if combine_type == 'i':
        combined_times = inf_time
    if combine_type == 'pi':
        combined_times = pre_time + inf_time
    if combine_type == 'ip':
        combined_times = inf_time + post_time
    if combine_type == 'pip':
        combined_times = pre_time + inf_time + post_time

    return combined_times


def calculate_acc(results, dataset_type='ImageNet', assets_path=pathlib.Path('assets'), threshold=None, times=None):
    assert dataset_type in dataset_choices, f"Wrong Type of Dataset: {dataset_type}"

    if dataset_type == 'ImageNet':
        # times should be array-like object
        if len(results) == 0:
            return float("NaN"), float("NaN")

        total = 0
        top5_correct = 0
        top1_correct = 0
        for index, instance in enumerate(results):
            truth_index = instance['image_id']
            top5_index = instance['result']['top5_class_indices']
            top1_index = top5_index[0]
            total += 1
            cur_top5_is_correct = 1 if truth_index in set(top5_index) else 0
            cur_top1_is_correct = 1 if truth_index == top1_index else 0
            if threshold is None:
                top5_correct += cur_top5_is_correct
                top1_correct += cur_top1_is_correct
            else:
                if times[index] < threshold:
                    top5_correct += cur_top5_is_correct
                    top1_correct += cur_top1_is_correct
                else:
                    pass

        top5_acc = top5_correct / total
        top1_acc = top1_correct / total
        return top5_acc, top1_acc

    if dataset_type == 'MMLU':
        # times should be a dict and each element is an array-like object
        if len(results) == 0:
            return dict(NaN=float("NaN")), float("NaN")

        task_stats = dict()
        task_acc = dict()
        for index, instance in enumerate(results):
            pred_answer = instance['pred_answer']
            gold_answer = instance['gold_answer']
            task = instance['task']

            task_stat = task_stats.get(task, (0, 0))
            task_correct = task_stat[0]
            task_total = task_stat[1] + 1
            task_is_correct = 1 if pred_answer == gold_answer else 0
            if threshold is None:
                task_correct = task_correct + task_is_correct
            else:
                if times[index] < threshold:
                    task_correct = task_correct + task_is_correct
                else:
                    pass
            task_stats[task] = (task_correct, task_total)
        correct = 0
        total = 0
        for task in task_stats.keys():
            task_correct = task_stats[task][0]
            task_total = task_stats[task][1]

            correct += task_correct
            total += task_total
            task_acc[task] = task_correct / task_total

        acc = correct / total
        return task_acc, acc


def calculate_map(results, dataset_type='COCO', assets_path=pathlib.Path('assets'), threshold=None, times=None, no_print=False):
    assert dataset_type in dataset_choices, f"Wrong Type of Dataset: {dataset_type}"

    def coco_json2numpy(json_results):
        numpy_results = numpy.ndarray((len(json_results), 7))

        for index, json_result in enumerate(json_results):
            numpy_results[index][0] = json_result['image_id']
            numpy_results[index][1] = json_result['bbox'][0]
            numpy_results[index][2] = json_result['bbox'][1]
            numpy_results[index][3] = json_result['bbox'][2]
            numpy_results[index][4] = json_result['bbox'][3]
            numpy_results[index][5] = json_result['score']
            numpy_results[index][6] = json_result['category_id']

        return numpy_results


    def remove_instances(results, threshold, times):
        if threshold is None:
            return results
        clean_results = list()
        for instance, time in zip(results, times):
            if time < threshold:
                clean_results.append(instance)
            else:
                pass

        return clean_results

    def expand_results(results):
        expanded_results = list()
        for instance in results:
            expanded_results.extend(instance['result'])
        return expanded_results


    if dataset_type == 'COCO':
        # times should be array-like object
        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
        except Exception:
            raise ImportError("Can not import COCO or COCOeval from pycocotools, please install the missing package by using \'pip install pycocotools\' ") from Exception

        if len(results) == 0:
            return [float("NaN"), ]

        results = remove_instances(results, threshold, times)
        results = expand_results(results)
        results = coco_json2numpy(results)
        if len(results) == 0:
            return [0, ]

        if no_print:
            original_stdout = sys.stdout
            sys.stdout = io.StringIO()
        anno_path = assets_path.joinpath("COCO", "annotations", "instances_val2017.json")
        coco_gt = COCO(anno_path)

        coco_dt = coco_gt.loadRes(results)
        cocoEval = COCOeval(coco_gt, coco_dt, iouType='bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        if no_print:
            sys.stdout = original_stdout
        return cocoEval.stats


def calculate_wf1(results, dataset_type='MELD', assets_path=pathlib.Path('assets'), threshold=None, times=None):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate Quality')
    parser.add_argument('-d', '--data-dir', type=str, required=True)
    parser.add_argument('-n', '--data-filename', type=str, required=True)

    parser.add_argument('-r', '--run-index', type=int, default=None)
    parser.add_argument('-t', '--threshold', type=float, default=None)
    parser.add_argument('-a', '--assets-path', type=str, default='assets')

    parser.add_argument('-q', '--quality-type', type=str, default='acc', choices=quality_choices)
    parser.add_argument('-s', '--dataset-type', type=str, default='ImageNet', choices=dataset_choices)
    parser.add_argument('-c', '--combine-type', type=str, default='i', choices=combine_choices)
    arguments = parser.parse_args()

    quality_type = arguments.quality_type
    dataset_type = arguments.dataset_type
    combine_type = arguments.combine_type

    data_dir = pathlib.Path(arguments.data_dir)
    data_filename = arguments.data_filename
    assert data_dir.is_dir(), f"No Such Data Dir: {data_dir}"

    extracted_data = extract_data(data_dir, data_filename, dataset_type)

    run_index = arguments.run_index
    threshold = arguments.threshold
    if threshold is None:
        print(f"Calculating Quality without setting threshold.")
        times = None
    else:
        print(f"Calculating Quality with threshold: {threshold}.")
        times = combine_times(extracted_data['other_results'], combine_type)
        if times.shape[0] == 0:
            threshold = None
            times = None
        else:
            times = times[:, run_index]

    assets_path = pathlib.Path(arguments.assets_path)
    assert assets_path.is_dir(), f"No Such assets Dir: {assets_path}"

    quality_calculate = globals()['calculate_' + quality_type]

    quality = quality_calculate(extracted_data['main_results'], dataset_type, assets_path=assets_path, threshold=threshold, times=times)

    if quality_type == 'acc' and dataset_type == 'ImageNet':
        print(f"Top-5 Accuracy = {quality[0] * 100:.4f} %")
        print(f"Top-1 Accuracy = {quality[1] * 100:.4f} %")

    if quality_type == 'acc' and dataset_type == 'MMLU':
        print(f"Task Accuracies:")
        for task in quality[0].keys():
            task_name = ' '.join([task_sub.capitalize() for task_sub in task.split('_')])
            print(f"{task_name} = {quality[0][task] * 100:.4f} %")
        print(f"Total Accuracy = {quality[1] * 100:.4f} %")

    if quality_type == 'map' and dataset_type == 'COCO':
        print(f"Average Precision (AP) @[ IoU=0.5:0.95 | area=all | maxDets=100 ] = {quality[0] * 100:.4f} %")