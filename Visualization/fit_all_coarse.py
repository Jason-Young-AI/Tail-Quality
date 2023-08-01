import time
import numpy
import pickle
import pathlib
import argparse

import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV

from calculate_quality import combine_times, calculate_stat
from extract_data import extract_data
from constant import dataset_choices, combine_choices, rm_outs_choices, kernel_choices, level_choices, fit_choices


def remove_outliers(new_times, new_stats, outliers_mode):
    instance_number, run_number = new_times.shape
    outlier_flags = numpy.array([[False for _ in range(run_number)] for i in range(instance_number)])
    if outliers_mode == 'quantile':
        q1s = new_stats['q1s'].reshape(-1, 1)
        q3s = new_stats['q3s'].reshape(-1, 1)
        iqr = q3s - q1s
        lower_outlier_flags = new_times < (q1s - 1.5 * iqr)
        upper_outlier_flags = (q3s + 1.5 * iqr < new_times)
        outlier_flags = lower_outlier_flags | upper_outlier_flags
    if outliers_mode == 'guassian':
        outlier_flags = numpy.absolute(new_times - new_stats['avgs'].reshape(-1, 1)) > 3 * new_stats['stds'].reshape(-1, 1)
    if outliers_mode == 'median':
        meds = new_stats['meds'].reshape(-1, 1)
        mads = numpy.median(numpy.absolute(new_times - meds), axis=-1)
        outlier_flags = numpy.absolute(new_times - meds) > 3 * mads

    return outlier_flags


def fit(ins_times, fit_type='gmm'):
    if fit_type == 'gmm':
        ns = list(range(1, 11))
        models = [GaussianMixture(n_components=n, covariance_type='full', random_state=0).fit(ins_times) for n in ns]
        bics = [model.bic(ins_times) for model in models]
        scores = [model.score(ins_times) for model in models]
        id = numpy.argmin(bics)
        best_model = models[id]
        best_p = ns[id]
        score = scores[id]
    if fit_type == 'kde':
        params = {'bandwidth': numpy.logspace(-4, 0, 30)}
        grid = GridSearchCV(KernelDensity(kernel='gaussian'), params, cv=5)
        grid.fit(ins_times)
        best_model = grid.best_estimator_
        best_p = grid.best_params_['bandwidth']
        score = best_model.score(ins_times)

    return best_model, best_p, score


def get_gaussian(ins_imes):
    pass


def fit_all(times, min_ns, np_fit, fit_type):
    tic = time.perf_counter()
    all_dist = list()
    for index, (ins_times, min_n, ins_np_fit) in enumerate(zip(times, min_ns, np_fit)):
        if numpy.sum(ins_np_fit) != 0:
            print(f"No.{index} Fitting.")
            ins_times = ins_times[:min_n].reshape(-1, 1)
            all_dist.append(fit(ins_times, fit_type))
        else:
            print(f"No.{index} Gaussian.")
            all_dist.append(get_gaussian(ins_times))
            
    toc = time.perf_counter()
    print(f"Total time consume: {toc-tic:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate Quality')
    parser.add_argument('-d', '--data-dir', type=str, required=True)
    parser.add_argument('-n', '--data-filename', type=str, required=True)

    parser.add_argument('-f', '--fit-npz-path', type=str, default=None)

    parser.add_argument('-m', '--fit-type', type=str, default='gmm', choices=fit_choices)

    parser.add_argument('-s', '--dataset-type', type=str, default='ImageNet', choices=dataset_choices)
    parser.add_argument('-c', '--combine-type', type=str, default='i', choices=combine_choices)
    parser.add_argument('-r', '--rm-outs-type', type=str, default='none', choices=rm_outs_choices)

    parser.add_argument('--instance-index', type=int, default=-1)
    arguments = parser.parse_args()

    fit_type = arguments.fit_type

    dataset_type = arguments.dataset_type
    combine_type = arguments.combine_type
    rm_outs_type = arguments.rm_outs_type

    data_dir = pathlib.Path(arguments.data_dir)
    data_filename = arguments.data_filename
    assert data_dir.is_dir(), f"No Such Data Dir: {data_dir}"

    extracted_data = extract_data(data_dir, data_filename, dataset_type)
    combined_times = combine_times(extracted_data['other_results'], combine_type)

    if dataset_type == 'COCO':
        batch_image_sizes = extracted_data['other_results']['batch_image_sizes']
        numpy.savez('coco.npz', times=combined_times, batch_image_sizes=batch_image_sizes)
    if dataset_type == 'ImageNet':
        batch_image_sizes = extracted_data['other_results']['batch_image_sizes']
        origin_image_sizes = extracted_data['other_results']['origin_image_sizes']
        numpy.savez('imagenet.npz', times=combined_times, batch_image_sizes=batch_image_sizes, origin_image_sizes=origin_image_sizes)
    if dataset_type == 'MMLU':
        token_lengths = extracted_data['other_results']['token_lengths']
        numpy.savez('mmlu.npz', times=combined_times, token_lengths=token_lengths)