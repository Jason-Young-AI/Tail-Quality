import time
import scipy
import numpy
import pickle
import pathlib
import argparse

import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from scipy.spatial.distance import jensenshannon
from sklearn.linear_model import LinearRegression

from calculate_quality import combine_times, calculate_stat
from extract_data import extract_data, get_main_record
from constant import dataset_choices, combine_choices, rm_outs_choices, kernel_choices, level_choices, fit_choices, fit_map


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


def gmm_aic(n_components, ins_times):
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(ins_times)
    return gmm.aic(ins_times)


def kde_aic(bandwidth, ins_times):
    kde = KernelDensity(bandwidth=bandwidth)
    kde.fit(ins_times)
    log_likelihood = kde.score(ins_times)
    num_params = 2  # KDE has two parameters: bandwidth and kernel
    num_samples = ins_times.shape[0]
    return -2 * log_likelihood + 2 * num_params + (2 * num_params * (num_params + 1)) / (num_samples - num_params - 1)


def fit(ins_times, fit_type='kde'):
    ins_times = ins_times.reshape(-1, 1)
    if fit_type == 'kde':
        bandwidth_grid = [0.005, 0.01, 0.03, 0.07, 0.1]
        best_bandwidth  = min(bandwidth_grid, key=lambda x: kde_aic(x, ins_times))
        model = KernelDensity(bandwidth=best_bandwidth).fit(ins_times)
    if fit_type == 'gmm':
        n_components_grid = [2, 3, 4, 5, 6]
        best_n_components = min(n_components_grid, key=lambda x: gmm_aic(x, ins_times))
        model = GaussianMixture(n_components=best_n_components).fit(ins_times)
    return model


def check_all_jsdis(check_model, models, ins_times):
    js_dis = list()
    x = numpy.linspace(ins_times.min(), ins_times.max(), 1000).reshape(-1, 1)
    epsilon = 1e-8
    for model in models:
        js_dis.append(jensenshannon(numpy.exp(check_model.score_samples(x))+epsilon, numpy.exp(model.score_samples(x))+epsilon))

    if len(js_dis) == 0:
        js_dis = [1,]

    return js_dis


def get_gaussian(ins_imes):
    stat = calculate_stat(ins_times)
    model = scipy.stats.norm(loc=stat['avgs'][0], scale=stat['stds'][0])
    return model


def get_model_pdf(x, model, model_type):
    if model_type in ['kde', 'gmm']:
        ps = numpy.exp(model.score_samples(x))
    if model_type == 'gau':
        ps = model.pdf(x)

    return ps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate Quality')
    parser.add_argument('-v', '--save-dir', type=str, required=True)

    parser.add_argument('-d', '--data-dir', type=str, required=True)
    parser.add_argument('-n', '--data-filename', type=str, required=True)
    parser.add_argument('-b', '--batch-size', type=int, required=True)

    parser.add_argument('-m', '--fit-type', type=str, default='kde', choices=fit_choices)

    parser.add_argument('-s', '--dataset-type', type=str, default='ImageNet', choices=dataset_choices)
    parser.add_argument('-c', '--combine-type', type=str, default='i', choices=combine_choices)

    parser.add_argument('--instance-indices', type=int, nargs='+', default=[-1,])
    arguments = parser.parse_args()

    save_dir = pathlib.Path(arguments.save_dir)

    if not save_dir.is_dir():
        print(f"No Imgs Save Dir Exists: {save_dir}, now creating it.")
        save_dir.mkdir(parents=True, exist_ok=True)

    fit_type = arguments.fit_type

    dataset_type = arguments.dataset_type
    combine_type = arguments.combine_type

    data_dir = pathlib.Path(arguments.data_dir)
    data_filename = arguments.data_filename
    assert data_dir.is_dir(), f"No Such Data Dir: {data_dir}"

    extracted_data = extract_data(data_dir, data_filename, dataset_type)
    combined_times = combine_times(extracted_data['other_results'], combine_type)
    combined_times = get_main_record(combined_times, arguments.batch_size)

    models = list()
    fig, axes = plt.subplots(1, 1, figsize=(10, 10))
    i_str = ""
    all_ins_times = combined_times[arguments.instance_indices]
    avg_times = calculate_stat(all_ins_times)['avgs']
    min_avg_time = numpy.min(avg_times)
    xs = numpy.linspace(numpy.min(all_ins_times), numpy.max(all_ins_times), num=1000)

    ax = axes
    for order, (index, ins_times, avg_time) in enumerate(zip(arguments.instance_indices, all_ins_times, avg_times)):
        ins_times = all_ins_times[order].reshape(-1, 1)
        ins_times = ins_times - avg_time + min_avg_time
        best_model = fit(ins_times, fit_type)
        models.append(best_model)
        this_xs = xs - avg_time + min_avg_time
        ys = get_model_pdf(this_xs.reshape(-1, 1), best_model, fit_type)

        ax.hist(ins_times, bins=30, alpha=0.5, density=True, label=f'Image {index}')
        ax.plot(xs, ys, color='xkcd:azure', label=f'{fit_map[fit_type]} {index}')
        i_str += f"_{index}"
    
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
    for i, line in enumerate(ax.lines):
        if i == len(arguments.instance_indices):
            break
        line.set_color(colors[i])

    ax.set_xlabel('Time')
    ax.set_ylabel('Density')
    ax.legend()

    print(f' - JS-Dis: ')
    for order, (index, model) in enumerate(zip(arguments.instance_indices, models)):
        print(f'  Ins:{index} - {check_all_jsdis(model, models, all_ins_times[order])}')

    figpath = save_dir.joinpath(f'fit{i_str}.pdf')
    fig.savefig(figpath)
    print(f' - Fig Exported: {figpath}')
