import time
import scipy
import numpy
import pickle
import pathlib
import argparse

import matplotlib as mpl
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
from heat import get_model_pdf, fit_all, fit , check_all_jsdis


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


if __name__ == "__main__":
    seed_value = 3456
    numpy.random.seed(seed_value)

    parser = argparse.ArgumentParser(description='Calculate Quality')
    parser.add_argument('-v', '--save-dir', type=str, required=True)

    arguments = parser.parse_args()

    save_dir = pathlib.Path(arguments.save_dir)

    if not save_dir.is_dir():
        print(f"No Imgs Save Dir Exists: {save_dir}, now creating it.")
        save_dir.mkdir(parents=True, exist_ok=True)

    combine_type = 'i'

    data_dirs = [
        pathlib.Path("../Results/Raw/GeForce_DETR_TensorFlow_val_bsz1/"),
        pathlib.Path("../Results/Raw/TITAN_V_DETR_TensorFlow_val_bsz1/"),
        pathlib.Path("../Results/Raw/P100_DETR_TensorFlow_val_bsz1/"),
    ]
    labels = [
        "Server A",
        "Server B",
        "Server C",
    ]
    data_filename = "DETR_Run100"
    fit_type = 'kde'
    dataset_type = 'COCO'

    extracted_data = extract_data(data_dirs[0], data_filename, dataset_type)
    combined_times = combine_times(extracted_data['other_results'], combine_type)
    combined_times = get_main_record(combined_times, 1)

    batch_image_sizes = get_main_record(extracted_data['other_results']['batch_image_sizes'], 1)
    instance_sizes = numpy.prod(batch_image_sizes, axis=-1)

    bis_to_cts = dict()
    for index, (instance_size, combined_time) in enumerate(zip(instance_sizes, combined_times)):
        ct = bis_to_cts.get(instance_size, list())
        ct.append((combined_time, index))
        bis_to_cts[instance_size] = ct

    bis_to_count = dict()
    for bis in  bis_to_cts.keys():
        bis_to_count[bis] = len(bis_to_cts[bis])
    
    bis_to_count = list(bis_to_count.items())
    bis_to_count = sorted(bis_to_count, key=lambda x: x[1])[::-1]
    bis_to_count = bis_to_count[:20]
    ids = numpy.random.choice(len(bis_to_count), size=3, replace=False)
    d_local_times = numpy.array([bis_to_cts[bis_to_count[id][0]][0][0] for id in ids])
    d_local_ids = numpy.array([bis_to_cts[bis_to_count[id][0]][0][1] for id in ids])

    s_times = bis_to_cts[bis_to_count[ids[0]][0]]
    main_id = d_local_ids[0]

    ids = numpy.random.choice(len(s_times), size=2, replace=False)
    s_local_times = [s_times[id][0] for id in ids]
    s_local_ids = [s_times[id][1] for id in ids]

    out_times = [d_local_times[0], ]

    extracted_data = extract_data(data_dirs[1], data_filename, dataset_type)
    combined_times = combine_times(extracted_data['other_results'], combine_type)
    combined_times = get_main_record(combined_times, 1)
    out_times.append(combined_times[main_id])

    extracted_data = extract_data(data_dirs[2], data_filename, dataset_type)
    combined_times = combine_times(extracted_data['other_results'], combine_type)
    combined_times = get_main_record(combined_times, 1)
    out_times.append(combined_times[main_id])

    out_times = numpy.array(out_times)[:, :70]


    local_times = numpy.concatenate([s_local_times, d_local_times])[:, :70]
    local_ids = numpy.concatenate([s_local_ids, d_local_ids])



    mpl.rcParams['font.size'] = 14
    mpl.rcParams['axes.labelsize'] = 16
    mpl.rcParams['xtick.labelsize'] = 14
    mpl.rcParams['ytick.labelsize'] = 14
    mpl.rcParams['legend.fontsize'] = 14

    cmap = plt.get_cmap('viridis')
    colors = [cmap(i) for i in numpy.linspace(0, 1, 7)]

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    ax = axes[0]
    tax = ax.twinx()

    xs = numpy.linspace(numpy.min(local_times), numpy.max(local_times), num=1000)
    for order, (ins_times, id) in enumerate(zip(local_times, local_ids)):
        # xs = numpy.linspace(numpy.min(ins_times), numpy.max(ins_times), num=1000)
        best_model = fit(ins_times, fit_type)
        ys = get_model_pdf(xs.reshape(-1, 1), best_model, fit_type)

        ax.hist(ins_times, bins=30, alpha=0.5, color=colors[order])
        tax.plot(xs, ys, label=f'# {id} ({batch_image_sizes[id][0]} x {batch_image_sizes[id][1]}) (Server A)', color=colors[order])
    
    ax.set_xlabel('Inference Time (Seconds)')
    ax.set_ylabel('Frequency')
    tax.legend()

    ax = axes[1]
    tax = ax.twinx()

    xs = numpy.linspace(numpy.min(out_times), numpy.max(out_times), num=1000)
    for order, ins_times in enumerate(out_times):
        # xs = numpy.linspace(numpy.min(ins_times), numpy.max(ins_times), num=1000)
        best_model = fit(ins_times, fit_type)
        ys = get_model_pdf(xs.reshape(-1, 1), best_model, fit_type)
        area = scipy.integrate.simps(ys, xs)
        print(area)

        ax.hist(ins_times, bins=30, alpha=0.5, color=colors[order+4])
        tax.plot(xs, ys, label=f'#{local_ids[0]} ({labels[order]})', color=colors[order+4])

    ax.set_xlabel('Inference Time (Seconds)')
    tax.set_ylabel('Probability Density')
    tax.legend()
    plt.tight_layout()


    figpath = save_dir.joinpath(f'fit.pdf')
    fig.savefig(figpath)
    print(f' - Fig Exported: {figpath}')
