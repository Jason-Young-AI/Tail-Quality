import re
import json
import numpy
import pathlib
import argparse

from constant import dataset_choices


def get_main_record(xs, batch_size):
    main_xs = list()
    for i, x in enumerate(xs):
        if i % batch_size == 0:
            main_xs.append(x)
    if isinstance(xs, numpy.ndarray):
        return numpy.array(main_xs)
    return main_xs


def filename_match_pattern(filename, pattern):
    regex = re.compile(pattern)
    return bool(regex.match(filename))


def extract_data_of_MMLU(results_filepaths, perform_filepaths):
    main_results_filepath = results_filepaths[0]
    print(f" + Only extract results from one file: \'{main_results_filepath}\' ...")
    with open(main_results_filepath) as main_results_file:
        results = json.load(main_results_file)

    main_results = list()

    tasks = list()
    task_offsets = list()
    token_lengths = list()
    question_numbers = list()

    task_offset = 0
    for task in results.keys():
        tasks.append(task)
        task_offsets.append(task_offset)
        for pred_answer, gold_answer, token_length, question_number in zip(results[task]['pred_answers'], results[task]['gold_answers'], results[task]['token_lengths'], results[task]['question_numbers']):
            main_results.append(dict(
                pred_answer = pred_answer,
                gold_answer = gold_answer,
                task = task,
            ))
            token_lengths.append(token_length)
            question_numbers.append(question_number)
            task_offset += 1

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
        for task in tasks:
            for instance in perform[task]['pred_times']:
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
        main_results = main_results, # list(dict(pred_answers: list(), gold_answers: list()))
        other_results = dict(
            tasks = tasks, # list()
            task_offsets = task_offsets, # list()
            token_lengths = token_lengths, # list()
            question_numbers = question_numbers, # list()
            inference_times = inference_times, # task: numpy.array[total_question_number * run_number]
            preprocess_times = preprocess_times, # task: numpy.array[total_question_number * run_number]
            postprocess_times = postprocess_times, # task: numpy.array[total_question_number * run_number]
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

    if len(results_filepaths) == 0:# or len(perform_filepaths) == 0:
        return None

    extract_data_by_dst = globals()['extract_data_of_' + dataset_type]
    extracted_data = extract_data_by_dst(results_filepaths, perform_filepaths)
    return extracted_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract Data')
    parser.add_argument('-d', '--data-dir', type=str, required=True)
    parser.add_argument('-n', '--data-filename', type=str, required=True)
    #parser.add_argument('-p', '--data-npz-path', type=str, default=None)

    parser.add_argument('-t', '--dataset-type', type=str, default='ImageNet', choices=dataset_choices)
    arguments = parser.parse_args()

    dataset_type = arguments.dataset_type

    data_dir = pathlib.Path(arguments.data_dir)
    data_filename = arguments.data_filename
    assert data_dir.is_dir(), f"No Such Data Dir: {data_dir}"

    extracted_data = extract_data(data_dir, data_filename, dataset_type)

    #if extracted_data is None:
    #    print(f" ! No Data Extracted")
    #else:
    #    if arguments.data_npz_path is None:
    #        print(" ! Do not save data to disk.")
    #    else:
    #        data_npz_path = pathlib.Path(arguments.data_npz_path)
    #        data_npz_path = data_npz_path.with_suffix('.npz')
    #        print(f" + Saving data into \'{data_npz_path}\' ...")
    #        numpy.savez(data_npz_path, **extracted_data)
    #        print(f" - Saved.")