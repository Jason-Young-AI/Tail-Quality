import json
import numpy
import pathlib
import argparse

quality_choices = ['acc', 'map', 'wf1']
dataset_choices = ['ImageNet', 'MSCOCO', 'MMLU']
combine_choices = ['i', 'ip', 'pi', 'pip'] # inf, inf->post, pre->inf, pre->inf->post


def combine_ImageNet_times(times, combine_type):
    assert combine_type in combine_choices, f"Wrong Type of Combine Type: {combine_type}"
    combined_times = list()
    for time_dict in times:
        inf_time = time_dict['inference_time']
        pre_time = time_dict['preprocess_time']
        post_time = time_dict['postprocess_time']
        if combine_type == 'i':
            combined_time = inf_time
        if combine_type == 'pi':
            combined_time = pre_time + inf_time
        if combine_type == 'ip':
            combined_time = inf_time + post_time
        if combine_type == 'pip':
            combined_time = pre_time + inf_time + post_time
        combined_times.append(combined_time)

    return combined_times


def combine_MMLU_times(times, combine_type):
    assert combine_type in combine_choices, f"Wrong Type of Combine Type: {combine_type}"
    combined_times = dict()
    for task in times.keys():
        combined_times[task] = list()
        for time_dict in times[task]['pred_times']:
            inf_time = time_dict['inference_time']
            pre_time = time_dict['preprocess_time']
            post_time = time_dict['postprocess_time']
            if combine_type == 'i':
                combined_time = inf_time
            if combine_type == 'pi':
                combined_time = pre_time + inf_time
            if combine_type == 'ip':
                combined_time = inf_time + post_time
            if combine_type == 'pip':
                combined_time = pre_time + inf_time + post_time
            combined_times[task].append(combined_time)

    return combined_times


def combine_MSCOCO_times(times, combine_type):
    assert combine_type in combine_choices, f"Wrong Type of Combine Type: {combine_type}"
    combined_times = list()
    for time_dict in times:
        inf_time = time_dict['inference_time']
        pre_time = time_dict['preprocess_time']
        post_time = time_dict['postprocess_time']
        if combine_type == 'i':
            combined_time = inf_time
        if combine_type == 'pi':
            combined_time = pre_time + inf_time
        if combine_type == 'ip':
            combined_time = inf_time + post_time
        if combine_type == 'pip':
            combined_time = pre_time + inf_time + post_time
        combined_times.extend([combined_time,] * 100) # maxDets is set to 100 as default.

    return combined_times


def calculate_acc(results, dataset_type='ImageNet', assets_path=pathlib.Path('assets'), threshold=None, times=None):
    assert dataset_type in dataset_choices, f"Wrong Type of Dataset: {dataset_type}"

    if dataset_type == 'ImageNet':
        if len(results) == 0:
            return float("NaN"), float("NaN")

        truth_indices = list()
        top5_indices = list()
        top1_indices = list()
        for instance in results:
            truth_index = instance['image_id']
            top5_index = instance['result']['top5_class_indices']
            top1_index = top5_index[0]
            truth_indices.append(truth_index)
            top5_indices.append(top5_index)
            top1_indices.append(top1_index)
        total = 0
        top5_correct = 0
        top1_correct = 0
        for index, (truth_index, top5_index, top1_index) in enumerate(zip(truth_indices, top5_indices, top1_indices)):
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
        if len(results) == 0:
            return float("NaN")

        task_totals = list()
        task_corrects = list()
        task_acc = dict()
        for task in results.keys():
            pred_answers = results[task]['pred_answers']
            gold_answers = results[task]['gold_answers']

            task_total = len(gold_answers)
            task_correct = 0
            for index, (pred_answer, gold_answer) in enumerate(zip(pred_answers, gold_answers)):
                cur_task_is_correct = 1 if pred_answer == gold_answer else 0
                if threshold is None:
                    task_correct += cur_task_is_correct
                else:
                    if times[task][index] < threshold:
                        task_correct += cur_task_is_correct
                    else:
                        pass
            task_totals.append(task_total)
            task_corrects.append(task_correct)
            task_acc[task] = task_correct / task_total

        total = numpy.sum(task_totals)
        correct = numpy.sum(task_corrects)
        acc = correct / total
        return task_acc, acc


def calculate_map(results, dataset_type='ImageNet', assets_path=pathlib.Path('assets'), threshold=None, times=None):
    assert dataset_type in dataset_choices, f"Wrong Type of Dataset: {dataset_type}"

    anno_path = assets_path.joinpath("annotations", "instances_val2017.json")

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


    if dataset_type == 'MSCOCO':
        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
        except Exception:
            raise ImportError("Can not import COCO or COCOeval from pycocotools, please install the missing package by using \'pip install pycocotools\' ") from Exception

        coco_gt = COCO(anno_path)

        results = remove_instances(results, threshold, times)
        results = coco_json2numpy(results)
        if len(results) == 0:
            return 0

        coco_dt = coco_gt.loadRes(results)
        cocoEval = COCOeval(coco_gt, coco_dt, iouType='bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        return cocoEval.stats


def calculate_wf1(results, dataset_type='ImageNet', assets_path=pathlib.Path('assets'), threshold=None, times=None):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate Quality')
    parser.add_argument('--results-path', type=str, required=True)
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--times-path', type=str, default=None)
    parser.add_argument('--assets-path', type=str, default='assets')
    parser.add_argument('--quality-type', type=str, default='acc', choices=quality_choices)
    parser.add_argument('--dataset-type', type=str, default='ImageNet', choices=dataset_choices)
    parser.add_argument('--combine-type', type=str, default='i', choices=combine_choices)
    arguments = parser.parse_args()

    results_path = pathlib.Path(arguments.results_path)

    assets_path = pathlib.Path(arguments.assets_path)
    quality_type = arguments.quality_type
    dataset_type = arguments.dataset_type

    assert results_path.is_file(), f"No Such Results File: {results_path}"

    assert assets_path.is_dir(), f"No Such assets Dir: {assets_path}"
    assert quality_type in quality_choices, f"No Such Results File: {quality_type}"
    assert dataset_type in dataset_choices, f"No Such Results File: {dataset_type}"

    threshold = arguments.threshold
    if threshold is None:
        print(f"Calculating Quality without setting threshold.")
        times = None
    else:
        print(f"Calculating Quality with threshold: {threshold}.")
        times_path = pathlib.Path(arguments.times_path)
        assert times_path.is_file(), f"No Such times File: {times_path}"
        times = json.load(open(times_path))
        combine_times = globals()['combine_' + dataset_type + '_times']
        times = combine_times(times, arguments.combine_type)

    results = json.load(open(results_path))

    quality_calculate = globals()['calculate_' + quality_type]

    quality = quality_calculate(results, dataset_type, assets_path=assets_path, threshold=threshold, times=times)

    if quality_type == 'acc' and dataset_type == 'ImageNet':
        print(f"Top-5 Accuracy = {quality[0] * 100:.4f} %")
        print(f"Top-1 Accuracy = {quality[1] * 100:.4f} %")

    if quality_type == 'acc' and dataset_type == 'MMLU':
        print(f"Task Accuracies:")
        for task in quality[0].keys():
            task_name = ' '.join([task_sub.capitalize() for task_sub in task.split('_')])
            print(f"{task_name} = {quality[0][task] * 100:.4f} %")
        print(f"Total Accuracy = {quality[1] * 100:.4f} %")

    if quality_type == 'map' and dataset_type == 'MSCOCO':
        print(f"Average Precision (AP) @[ IoU=0.5:0.95 | area=all | maxDets=100 ] = {quality[0] * 100:.4f} %")