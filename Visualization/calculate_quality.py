import json
import numpy
import pathlib
import argparse

quality_choices = ['acc', 'map', 'wf1']
dataset_choices = ['ImageNet', 'MSCOCO', 'MMLU']


def calculate_acc(results, dataset_type='ImageNet'):
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
        for truth_index, top5_index, top1_index in zip(truth_indices, top5_indices, top1_indices):
            total += 1
            top5_correct += 1 if truth_index in set(top5_index) else 0
            top1_correct += 1 if truth_index == top1_index else 0

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
            for pred_answer, gold_answer in zip(pred_answers, gold_answers):
                if pred_answer == gold_answer:
                    task_correct += 1
            task_totals.append(task_total)
            task_corrects.append(task_correct)
            task_acc[task] = task_correct / task_total

        total = numpy.sum(task_totals)
        correct = numpy.sum(task_corrects)
        acc = correct / total
        return task_acc, acc


def calculate_map(results):
    assert dataset_type in dataset_choices, f"Wrong Type of Dataset: {dataset_type}"

    if dataset_type == 'MSCOCO':


def calculate_wf1(results):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate Quality')
    parser.add_argument('--results-path', type=str, required=True)
    parser.add_argument('--quality-type', type=str, default='acc', choices=quality_choices)
    parser.add_argument('--dataset-type', type=str, default='ImageNet', choices=dataset_choices)
    arguments = parser.parse_args()

    results_path = arguments.results_path
    quality_type = arguments.quality_type
    dataset_type = arguments.dataset_type

    assert pathlib.Path(results_path).is_file(), f"No Such Results File: {results_path}"
    assert quality_type in quality_choices, f"No Such Results File: {quality_type}"
    assert dataset_type in dataset_choices, f"No Such Results File: {dataset_type}"

    results = json.load(open(results_path))

    quality_calculate = globals()['calculate_' + arguments.quality_type]

    quality = quality_calculate(results, dataset_type)

    if quality_type == 'acc' and dataset_type == 'ImageNet':
        print(f"Top-5 Accuracy = {quality[0] * 100:.4f} %")
        print(f"Top-1 Accuracy = {quality[1] * 100:.4f} %")

    if quality_type == 'acc' and dataset_type == 'MMLU':
        print(f"Task Accuracies:")
        for task in quality[0].keys():
            task_name = ' '.join([task_sub.capitalize() for task_sub in task.split('_')])
            print(f"{task_name} = {quality[0][task] * 100:.4f} %")
        print(f"Total Accuracy = {quality[1] * 100:.4f} %")