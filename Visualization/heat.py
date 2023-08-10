import time
import scipy
import numpy
import pickle
import pathlib
import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from scipy.spatial.distance import jensenshannon
from sklearn.linear_model import LinearRegression

from calculate_quality import combine_times, calculate_stat
from extract_data import extract_data, get_main_record
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


def get_model_pdf(x, model, model_type):
    if model_type in ['kde', 'gmm']:
        ps = numpy.exp(model.score_samples(x))
    if model_type == 'gau':
        ps = model.pdf(x)

    return ps


def check_all_jsdis(check_model, check_model_type, models, model_types, ins_times):
    js_dis = list()
    epsilon = 1e-8
    x = numpy.linspace(ins_times.min(), ins_times.max(), 1000).reshape(-1, 1)
    for model, model_type in zip(models, model_types):
        js_dis.append(pow(jensenshannon(get_model_pdf(x, check_model, check_model_type)+epsilon, get_model_pdf(x, model, model_type)+epsilon), 2))

    if len(js_dis) == 0:
        js_dis = [1,]

    return js_dis


def get_gaussian(ins_times):
    stat = calculate_stat(ins_times)
    model = scipy.stats.norm(loc=stat['avgs'][0], scale=stat['stds'][0])
    return model


def fit_all(times, min_ns, fit_type):
    tic = time.perf_counter()
    models = list()
    model_types = list()
    for index, (ins_times, min_n) in enumerate(zip(times, min_ns)):
        ins_times = ins_times[:min_n].reshape(-1, 1)
        print(f"No.{index+1} Fitting.")
        best_model = fit(ins_times, fit_type)
        models.append(best_model)
        model_types.append(fit_type)
            
    toc = time.perf_counter()
    print(f"Total time consume: {toc-tic:.2f}s")
    return models, model_types


if __name__ == "__main__":
    seed_value = 66
    numpy.random.seed(seed_value)

    parser = argparse.ArgumentParser(description='Calculate Quality')
    parser.add_argument('-v', '--save-dir', type=str, required=True)

    parser.add_argument('-d', '--data-dir', type=str, required=True)
    parser.add_argument('-n', '--data-filename', type=str, required=True)
    parser.add_argument('-u', '--check-npz-path', type=str, required=True)
    parser.add_argument('-b', '--batch-size', type=int, required=True)

    parser.add_argument('-m', '--fit-type', type=str, default='kde', choices=fit_choices)

    parser.add_argument('-s', '--dataset-type', type=str, default='ImageNet', choices=dataset_choices)
    parser.add_argument('-c', '--combine-type', type=str, default='i', choices=combine_choices)
    parser.add_argument('-r', '--rm-outs-type', type=str, default='none', choices=rm_outs_choices)

    arguments = parser.parse_args()

    save_dir = pathlib.Path(arguments.save_dir)

    if not save_dir.is_dir():
        print(f"No Imgs Save Dir Exists: {save_dir}, now creating it.")
        save_dir.mkdir(parents=True, exist_ok=True)

    fit_type = arguments.fit_type

    dataset_type = arguments.dataset_type
    combine_type = arguments.combine_type
    rm_outs_type = arguments.rm_outs_type

    data_dir = pathlib.Path(arguments.data_dir)
    data_filename = arguments.data_filename
    assert data_dir.is_dir(), f"No Such Data Dir: {data_dir}"

    extracted_data = extract_data(data_dir, data_filename, dataset_type)
    combined_times = combine_times(extracted_data['other_results'], combine_type)
    combined_times = get_main_record(combined_times, arguments.batch_size)
    if dataset_type == 'ImageNet':
        origin_image_sizes = get_main_record(extracted_data['other_results']['origin_image_sizes'], arguments.batch_size)
        instance_sizes = numpy.prod(origin_image_sizes, axis=-1)

    if dataset_type == 'COCO':
        batch_image_sizes = get_main_record(extracted_data['other_results']['batch_image_sizes'], arguments.batch_size)
        instance_sizes = numpy.prod(batch_image_sizes, axis=-1)

    if dataset_type == 'MMLU':
        token_lengths = get_main_record(extracted_data['other_results']['token_lengths'], arguments.batch_size)
        instance_sizes = numpy.array(token_lengths)


    instance_sizes, unique_indices = numpy.unique(instance_sizes, return_index=True)
    combined_times = combined_times[unique_indices]

    inum = len(instance_sizes)
    snum = min(len(instance_sizes), 100)
    print(inum)

    random_i = numpy.random.choice(inum, size=snum, replace=False)
    instance_sizes = instance_sizes[random_i]
    combined_times = combined_times[random_i]

    ids = numpy.argsort(instance_sizes)
    combined_times = combined_times[ids]

    min_ns = numpy.array([70 for _ in range(combined_times.shape[0])])

    models, model_types = fit_all(combined_times, min_ns, fit_type)

    all_jsdis = list()
    total_avg_jsdis = 0
    for index, (model, model_type) in enumerate(zip(models, model_types)):
        this_jsdis = numpy.array(check_all_jsdis(model, model_type, models, model_types, combined_times))
        #print(this_jsdis)
        all_jsdis.append(this_jsdis)
        print(instance_sizes[ids][index])
        #this_avg_jsdis = numpy.sum(this_jsdis) / (len(models) - 1)
        #total_avg_jsdis += this_avg_jsdis
        #print(f"No.{index+1} JS-Dis Calculated: Avg={this_avg_jsdis}; Avg_Time:{numpy.average(combined_times[index])}.")

    #total_avg_jsdis = total_avg_jsdis / len(models)
    #print(f"Total Average JS-Dis: {total_avg_jsdis}")

    all_jsdis = numpy.array(all_jsdis)

    mpl.rcParams['font.size'] = 14
    mpl.rcParams['axes.labelsize'] = 16
    mpl.rcParams['xtick.labelsize'] = 14
    mpl.rcParams['ytick.labelsize'] = 14
    mpl.rcParams['legend.fontsize'] = 14

    cmap = plt.get_cmap('viridis')

    fig, axes = plt.subplots(1, 1, figsize=(10, 10))
    ax = axes

    im = ax.imshow(all_jsdis, cmap=cmap, vmin=0, vmax=1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Jensen–Shannon Divergence (JSD)', rotation=270, labelpad=18)
    #cbar = fig.colorbar(ax, ticks=[0, 1])
    #cbar.ax.set_yticklabels(['0', '1'])
    ax.set_xlabel('Image Index (Sorted by Pixel Dimensions)')
    ax.set_ylabel('Image Index (Sorted by Pixel Dimensions)')
    plt.tight_layout()

    figpath = save_dir.joinpath(f'js_heat_map.pdf')
    fig.savefig(figpath)
    print(f' - Fig Exported: {figpath}')