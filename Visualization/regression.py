import sys
import time
import scipy
import numpy
import pickle
import pathlib
import argparse

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

from calculate_quality import combine_times, calculate_stat
from extract_data import extract_data, get_main_record
from constant import dataset_choices, combine_choices, rm_outs_choices


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate Quality')
    parser.add_argument('-v', '--save-dir', type=str, required=True)

    parser.add_argument('-d', '--data-dir', type=str, required=True)
    parser.add_argument('-n', '--data-filename', type=str, required=True)
    parser.add_argument('-b', '--batch-size', type=int, required=True)

    parser.add_argument('-s', '--dataset-type', type=str, default='ImageNet', choices=dataset_choices)
    parser.add_argument('-c', '--combine-type', type=str, default='i', choices=combine_choices)

    arguments = parser.parse_args()

    save_dir = pathlib.Path(arguments.save_dir)

    if not save_dir.is_dir():
        print(f"No Imgs Save Dir Exists: {save_dir}, now creating it.")
        save_dir.mkdir(parents=True, exist_ok=True)

    dataset_type = arguments.dataset_type
    combine_type = arguments.combine_type
    if dataset_type == 'ImageNet':
        print(f"ImageNet Not Support! Exit!")
        sys.exit(0)

    data_dir = pathlib.Path(arguments.data_dir)
    data_filename = arguments.data_filename
    assert data_dir.is_dir(), f"No Such Data Dir: {data_dir}"

    extracted_data = extract_data(data_dir, data_filename, dataset_type)
    combined_times = combine_times(extracted_data['other_results'], combine_type)
    combined_times = get_main_record(combined_times, arguments.batch_size)
    if dataset_type == 'COCO':
        batch_image_sizes = get_main_record(extracted_data['other_results']['batch_image_sizes'], arguments.batch_size)
        instance_sizes = numpy.prod(batch_image_sizes, axis=-1)

    if dataset_type == 'MMLU':
        token_lengths = get_main_record(extracted_data['other_results']['token_lengths'], arguments.batch_size)
        instance_sizes = numpy.array(token_lengths)

    times_stat = calculate_stat(combined_times)
    reg_avg = LinearRegression()
    reg_avg = reg_avg.fit(instance_sizes.reshape(-1, 1), times_stat['avgs'])
    avg_pred = reg_avg.predict(instance_sizes.reshape(-1, 1))
    avg_mse = mean_squared_error(times_stat['avgs'], avg_pred)
    avg_r2 = r2_score(times_stat['avgs'], avg_pred)
    print(f'Avg Reg: MSE={avg_mse} R2={avg_r2}')

    fig, axes = plt.subplots(1, 1, figsize=(10, 10))
    ax = axes

    ax.scatter(instance_sizes, times_stat['avgs'], color='blue', label='Original data')
    ax.plot(instance_sizes, avg_pred, color='red', label='Fitted line')
    ax.set_xlabel('Image Size')
    ax.set_ylabel('Average Time')

    figpath = save_dir.joinpath(f'reg_avg.pdf')
    fig.savefig(figpath)
    print(f' - Fig Exported: {figpath}')

    reg_var = LinearRegression()
    reg_var = reg_var.fit(instance_sizes.reshape(-1, 1), times_stat['vars'])
    var_pred = reg_var.predict(instance_sizes.reshape(-1, 1))
    var_mse = mean_squared_error(times_stat['vars'], var_pred)
    var_r2 = r2_score(times_stat['vars'], var_pred)
    print(f'Var Reg: MSE={var_mse} R2={var_r2}')

    fig, axes = plt.subplots(1, 1, figsize=(10, 10))
    ax = axes

    ax.scatter(instance_sizes, times_stat['vars'], color='blue', label='Original data')
    ax.plot(instance_sizes, var_pred, color='red', label='Fitted line')
    ax.set_xlabel('Image Size')
    ax.set_ylabel('Variation')

    figpath = save_dir.joinpath(f'reg_var.pdf')
    fig.savefig(figpath)
    print(f' - Fig Exported: {figpath}')