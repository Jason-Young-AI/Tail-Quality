import re
import json
import numpy
import pathlib
import argparse

from constant import dataset_choices


def filename_match_pattern(filename, pattern):
    regex = re.compile(pattern)
    return bool(regex.match(filename))


def extract_data_of_MMLU(results_filepaths, perform_filepaths):
    main_results_filepath = results_filepaths[0]
    print(f" + Only extract results from one file: \'{main_results_filepath}\' ...")
    with open(main_results_filepath) as main_results_file:
        results = json.load(main_results_file)

    main_results = dict()

    tasks = list()
    token_lengths = dict()
    question_numbers = dict()

    for task in results.keys():
        main_results[task] = dict(
            pred_answers = results[task]['pred_answers'],
            gold_answers = results[task]['gold_answers']
        )
        tasks.append(task)
        token_lengths[task] = list()
        question_numbers[task] = list()
        for token_length, question_number in zip(results[task]['token_lengths'], results[task]['question_numbers']):
            token_lengths[task].append(token_length)
            question_numbers[task].append(question_number)

    print(f" - Results Extracted.")

    inference_times = dict()
    preprocess_times = dict()
    postprocess_times = dict()

    for task in tasks:
        inference_times[task] = list()
        preprocess_times[task] = list()
        postprocess_times[task] = list()

    print(f" + Extracting perform from {len(perform_filepaths)} files ...")
    for perform_filepath in perform_filepaths:
        with open(perform_filepath) as perform_file:
            perform = json.load(perform_file)
        for task in perform.keys():
            inference_time = list()
            preprocess_time = list()
            postprocess_time = list()
            for instance in perform[task]['pred_times']:
                inference_time.append(instance['inference_time'])
                preprocess_time.append(instance['preprocess_time'])
                postprocess_time.append(instance['postprocess_time'])
            inference_times[task].append(inference_time)
            preprocess_times[task].append(preprocess_time)
            postprocess_times[task].append(postprocess_time)
    print(f" - Perform Extracted.")

    for task in tasks:
        inference_times[task] = numpy.array(inference_times[task]).transpose()
        preprocess_times[task] = numpy.array(preprocess_times[task]).transpose()
        postprocess_times[task] = numpy.array(postprocess_times[task]).transpose()

    return dict(
        main_results = main_results, # dict(task: {pred_answers: list(), gold_answers: list()})
        other_results = dict(
            tasks = tasks, # list()
            token_lengths = token_lengths, # dict(list())
            question_numbers = question_numbers, # dict(list())
            inference_times = inference_times, # dict(task: numpy.array[run_number * question_number])
            preprocess_times = preprocess_times, # dict(task: numpy.array[run_number * question_number])
            postprocess_times = postprocess_times, # dict(task: numpy.array[run_number * question_number])
        )
    )


def extract_data_of_COCO(results_filepaths, perform_filepaths):
    main_results_filepath = results_filepaths[0]
    print(f" + Only extract results from one file: \'{main_results_filepath}\' ...")
    with open(main_results_filepath) as main_results_file:
        results = json.load(main_results_file)

    main_results = list()
    image_ids = list()

    batch_image_sizes = list()
    origin_image_sizes = list()

    for instance in results:
        main_results.append(dict(
            result = instance['result']
        ))
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
        main_results = main_results, # list(dict(result: list(100 * dict(image_id: int, category_id: int, bbox: list(x,y,w,h), score: float))))
        other_results = dict(
            image_ids = image_ids, # list()
            batch_image_sizes = batch_image_sizes, # list([H, W])
            origin_image_sizes = origin_image_sizes, # list([H, W])
            inference_times = inference_times, # numpy.array[image_number, run_number]
            preprocess_times = preprocess_times, # numpy.array[image_number, run_number]
            postprocess_times = postprocess_times, # numpy.array[image_number, run_number]
        )
    )

def extract_data_of_ImageNet(results_filepaths, perform_filepaths):
    main_results_filepath = results_filepaths[0]
    print(f" + Only extract results from one file: \'{main_results_filepath}\' ...")
    with open(main_results_filepath) as main_results_file:
        results = json.load(main_results_file)

    main_results = list()
    image_ids = list()

    batch_image_sizes = list()
    origin_image_sizes = list()

    for instance in results:
        main_results.append(dict(
            image_id = instance['image_id'],
            result = instance['result']
        ))
        image_ids.append(instance['image_id'])
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
        main_results = main_results, # list(dict(image_id: int, result: dict(top5_probabilities: list(), top5_class_indices: list())))
        other_results = dict(
            image_ids = image_ids, # list()
            batch_image_sizes = batch_image_sizes, # list([H, W])
            origin_image_sizes = origin_image_sizes, # list([H, W])
            inference_times = inference_times,  # numpy.array[image_number, run_number]
            preprocess_times = preprocess_times,  # numpy.array[image_number, run_number] 
            postprocess_times = postprocess_times,  # numpy.array[image_number, run_number]
        )
    )


def extract_data(data_dir, data_filename, dataset_type):
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

    extract_data_by_dst = globals()['extract_data_of_' + dataset_type]
    extracted_data = extract_data_by_dst(results_filepaths, perform_filepaths)
    return extracted_data


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

    extracted_data = extract_data(data_dir, data_filename, dataset_type)

    if extracted_data is None:
        print(f" ! No Data Extracted")
    else:
        save_filepath = save_dir.joinpath(save_filename)
        print(f" + Saving data into \'{save_filepath}.npz\' ...")
        numpy.savez(save_filepath, **extracted_data)
        print(f" - Saved.")