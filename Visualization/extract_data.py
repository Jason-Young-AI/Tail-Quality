import os
import re
import json
import numpy
import pathlib
import argparse

dataset_choices = ['ImageNet', 'COCO']


def filename_match_pattern(filename, pattern):
    regex = re.compile(pattern)
    return bool(regex.match(filename))


def extract_data_of_coco(results_filepaths, perform_filepaths):
    main_results_filepath = results_filepaths[0]
    print(f" + Only extract results from one file: \'{main_results_filepath}\' ...")
    with open(main_results_filepath) as main_results_file:
        results = json.load(main_results_file)

    image_ids = list()

    origin_image_sizes = list()
    batch_image_sizes = list()

    for instance in results:
        image_ids.append(instance['result'][0]['image_id'])
        batch_image_sizes.append(instance['batch_image_size'])
        origin_image_sizes.append(instance['origin_image_size'])
    print(f" - Results Extracted.")

    inference_times = list()
    preprocess_times = list()
    postprocess_times = list()

    print(f" + Extracting perform from {len(perform_filepaths)} files ...")
    for perform_filepath in perform_filepaths:
        with open(perform_filepath) as perform_file:
            perform = json.load(perform_file)
        inference_time = list()
        preprocess_time = list()
        postprocess_time = list()
        for instance in perform:
            inference_time.append(instance['inference_time'])
            preprocess_time.append(instance['preprocess_time'])
            postprocess_time.append(instance['postprocess_time'])
        inference_times.append(inference_time)
        preprocess_times.append(preprocess_time)
        postprocess_times.append(postprocess_time)
    print(f" - Perform Extracted.")

    inference_times = numpy.array(inference_times).transpose()
    preprocess_times = numpy.array(preprocess_times).transpose()
    postprocess_times = numpy.array(postprocess_times).transpose()

    return dict(
        image_ids = image_ids,
        origin_image_sizes = origin_image_sizes,
        batch_image_sizes = batch_image_sizes,
        inference_times = inference_times,
        preprocess_times = preprocess_times,
        postprocess_times = postprocess_times,
    )

def extract_data_of_imagenet(results_filepaths, perform_filepaths):
    main_results_filepath = results_filepaths[0]
    print(f" + Only extract results from one file: \'{main_results_filepath}\' ...")
    with open(main_results_filepath) as main_results_file:
        results = json.load(main_results_file)

    image_ids = list()

    origin_image_sizes = list()
    batch_image_sizes = list()

    for instance in results:
        image_ids.append(instance['image_id'])
        origin_image_sizes.append(instance['origin_image_size'])
    print(f" - Results Extracted.")

    inference_times = list()
    preprocess_times = list()
    postprocess_times = list()

    print(f" + Extracting perform from {len(perform_filepaths)} files ...")
    for perform_filepath in perform_filepaths:
        with open(perform_filepath) as perform_file:
            perform = json.load(perform_file)
        inference_time = list()
        preprocess_time = list()
        postprocess_time = list()
        for instance in perform:
            inference_time.append(instance['inference_time'])
            preprocess_time.append(instance['preprocess_time'])
            postprocess_time.append(instance['postprocess_time'])
        inference_times.append(inference_time)
        preprocess_times.append(preprocess_time)
        postprocess_times.append(postprocess_time)
    print(f" - Perform Extracted.")

    inference_times = numpy.array(inference_times).transpose()
    preprocess_times = numpy.array(preprocess_times).transpose()
    postprocess_times = numpy.array(postprocess_times).transpose()

    return dict(
        image_ids = image_ids,
        origin_image_sizes = origin_image_sizes,
        batch_image_sizes = batch_image_sizes,
        inference_times = inference_times,
        preprocess_times = preprocess_times,
        postprocess_times = postprocess_times,
    )


def extract_data(data_dir, data_filename, dataset_type='ImageNet'):
    assert dataset_type in dataset_choices, f"Wrong Type of Dataset: {dataset_type}"

    results_filepaths = list()
    perform_filepaths = list()
    results_pattern = rf'{data_filename}\.main\.\d+'
    perform_pattern = rf'{data_filename}\.time\.\d+'
    for filepath in data_dir.iterdir():
        if filepath.is_file():
            if filename_match_pattern(filepath.name, results_pattern):
                results_filepaths.append(filepath)
            if filename_match_pattern(filepath.name, perform_pattern):
                perform_filepaths.append(filepath)

    if len(results_filepaths) == 0 or len(perform_filepaths) == 0:
        return None

    if dataset_type == 'ImageNet':
        data = extract_data_of_imagenet(results_filepaths, perform_filepaths)

    if dataset_type == 'COCO':
        data = extract_data_of_coco(results_filepaths, perform_filepaths)

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract Data')
    parser.add_argument('-d', '--data-dir', type=str, required=True)
    parser.add_argument('-n', '--data-filename', type=str, required=True)
    parser.add_argument('-s', '--save-dir', type=str, required=True)
    parser.add_argument('-f', '--save-filename', type=str, required=True)
    parser.add_argument('-t', '--dataset-type', type=str, default='ImageNet', choices=dataset_choices)
    arguments = parser.parse_args()

    data_dir = pathlib.Path(arguments.data_dir)
    data_filename = arguments.data_filename
    save_dir = pathlib.Path(arguments.save_dir)
    save_filename = arguments.save_filename
    dataset_type = arguments.dataset_type

    assert data_dir.is_dir(), f"No Such Data Dir: {data_dir}"
    assert dataset_type in dataset_choices, f"No Such Dataset Type: {dataset_type}"

    if not save_dir.is_dir():
        print(f"No Save Dir Exists: {save_dir}, now creating it.")
        save_dir.mkdir(parents=True, exist_ok=True)

    data = extract_data(data_dir, data_filename, dataset_type)

    if data is None:
        print(f" ! No Data Extracted")
    else:
        save_filepath = save_dir.joinpath(save_filename)
        print(f" + Saving data into \'{save_filepath}.npz\' ...")
        numpy.savez(save_filepath, **data)
        print(f" - Saved.")