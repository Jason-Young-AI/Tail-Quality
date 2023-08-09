import sys
import time
import scipy
import numpy
import pickle
import pathlib
import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

from calculate_quality import combine_times, calculate_stat
from extract_data import extract_data, get_main_record
from constant import dataset_choices, combine_choices, rm_outs_choices


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate Quality')
    parser.add_argument('-v', '--save-dir', type=str, required=True)

    parser.add_argument('-c', '--combine-type', type=str, default='i', choices=combine_choices)

    arguments = parser.parse_args()

    save_dir = pathlib.Path(arguments.save_dir)

    if not save_dir.is_dir():
        print(f"No Imgs Save Dir Exists: {save_dir}, now creating it.")
        save_dir.mkdir(parents=True, exist_ok=True)

    combine_type = arguments.combine_type

    data_dirs = [
        pathlib.Path("../Results/Raw/GeForce_DETR_TensorFlow_val_bsz1"),
        pathlib.Path("../Results/Raw/TITAN_V_DETR_TensorFlow_val_bsz1"),
        pathlib.Path("../Results/Raw/P100_DETR_TensorFlow_val_bsz1"),
        pathlib.Path("../Results/Raw/GeForce_DETR_PyTorch_val_bsz1"),
        pathlib.Path("../Results/Raw/TITAN_V_DETR_PyTorch_val_bsz1"),
        pathlib.Path("../Results/Raw/P100_DETR_PyTorch_val_bsz1"),
        pathlib.Path("../Results/Raw/GeForce_LLM_PyTorch_val_bsz1"),
        pathlib.Path("../Results/Raw/V100_LLM_PyTorch_val_bsz1"),
    ]

    labels = [
        "DETR - Server A - TensorFlow",
        "DETR - Server B - TensorFlow",
        "DETR - Server C - TensorFlow",
        "DETR - Server A - PyTorch",
        "DETR - Server B - PyTorch",
        "DETR - Server C - PyTorch",
        "Vicuna - Server A - PyTorch",
        "Vicuna - Server D - PyTorch",
    ]

    color_id = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
    ]

    l_data_filename = "LLM_Run100"
    d_data_filename = "DETR_Run100"

    mpl.rcParams['font.size'] = 14
    mpl.rcParams['axes.labelsize'] = 16
    mpl.rcParams['xtick.labelsize'] = 14
    mpl.rcParams['ytick.labelsize'] = 14
    mpl.rcParams['legend.fontsize'] = 14

    cmap = plt.get_cmap('viridis')
    colors = [cmap(i) for i in numpy.linspace(0, 1, 8)]
    #colors = [cmap(i) for i in numpy.linspace(0, 1, 6)]

    # cmap = plt.get_cmap('tab20b')
    # colors = [cmap(i) for i in range(0, 3*4, 4)]

    fig, axes = plt.subplots(1, 3, figsize=(30, 11))

    for id in range(2):
        ax = axes[id]
        for i, data_dir in enumerate(data_dirs[id*3:id*3+3]):
            edata = extract_data(data_dir, d_data_filename, 'COCO')

            times = combine_times(edata['other_results'], combine_type)
            times = get_main_record(times, 1)

            batch_image_sizes = get_main_record(edata['other_results']['batch_image_sizes'], 1)
            instance_sizes = numpy.prod(batch_image_sizes, axis=-1)

            times_stat = calculate_stat(times)

            reg = LinearRegression()
            reg = reg.fit(instance_sizes.reshape(-1, 1), times_stat['avgs'])
            pred = reg.predict(instance_sizes.reshape(-1, 1))
            mse = mean_squared_error(times_stat['avgs'], pred)
            r2 = r2_score(times_stat['avgs'], pred)
            print(f'Avg Reg: MSE={mse} R2={r2}')

            ax.scatter(instance_sizes, times_stat['avgs'], color=colors[color_id[id*3+i]], marker='x', s=5, alpha=0.3)
            ax.plot(instance_sizes, pred, color=colors[color_id[id*3+i]], linewidth=2)

        ax.set_xlabel('Image Size (Number of Pixels)')
        ax.set_ylabel('Average Inferece Time (Seconds)')
        ax.ticklabel_format(useMathText=True)
        if id == 0:
            ax.set_title('(a)', fontsize=18)
        if id == 1:
            ax.set_title('(b)', fontsize=18)

    ax = axes[2]
    for i, data_dir in enumerate(data_dirs[6:]):
        edata = extract_data(data_dir, l_data_filename, 'MMLU')

        times = combine_times(edata['other_results'], combine_type)
        times = get_main_record(times, 1)

        token_lengths = get_main_record(edata['other_results']['token_lengths'], 1)
        instance_sizes = numpy.array(token_lengths)

        times_stat = calculate_stat(times)

        reg = LinearRegression()
        reg = reg.fit(instance_sizes.reshape(-1, 1), times_stat['avgs'])
        pred = reg.predict(instance_sizes.reshape(-1, 1))
        mse = mean_squared_error(times_stat['avgs'], pred)
        r2 = r2_score(times_stat['avgs'], pred)
        print(f'Avg Reg: MSE={mse} R2={r2}')

        ax.scatter(instance_sizes, times_stat['avgs'], color=colors[color_id[6+i]], marker='x', s=5, alpha=0.3)
        ax.plot(instance_sizes, pred, color=colors[color_id[6+i]], linewidth=2)

    ax.set_xlabel('Prompt Length (Number of Tokens)')
    ax.set_ylabel('Average Inferece Time (Seconds)')
    ax.ticklabel_format(useMathText=True)
    ax.set_title('(c)', fontsize=18)

    legend_es = [Line2D([0], [0], color=colors[color_id[i]], marker='x', markersize=5, label=labels[i]) 
                   for i in range(len(labels))]

    fig.legend(handles=legend_es, loc='lower center', ncols=4, bbox_to_anchor=(0.5, 0.05))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.show()

    figpath = save_dir.joinpath(f'reg_avg.pdf')
    fig.savefig(figpath)
    print(f' - Fig Exported: {figpath}')