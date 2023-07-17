import json
import pathlib
import argparse

quality_choices = ['acc', 'map', 'wf1']
dataset_choices = ['ImageNet', 'MSCOCO']


def calculate_acc(results, dataset_type='ImageNet'):
    if len(results) == 0:
        return float("NaN"), float("NaN")

    truth_indices = list()
    top5_indices = list()
    top1_indices = list()
    assert dataset_type in dataset_choices, f"Wrong Type of Dataset: {dataset_type}"
    for instance in results:
        if dataset_type == 'ImageNet':
            truth_index = instance['image_id']
            top5_index = instance['result']['top5_class_indices']
            top1_index = top5_index[0]
            truth_indices.append(truth_index)
            top5_indices.append(top5_index)
            top1_indices.append(top1_index)

    if dataset_type == 'ImageNet':
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


def calculate_map(results):
    pass


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