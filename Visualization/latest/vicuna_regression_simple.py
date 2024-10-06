import numpy
import pickle
import pathlib
import argparse

import matplotlib.pyplot as plt

from typing import Any


def load_pickle(filepath: pathlib.Path) -> Any:
    info = None
    with open(filepath, 'rb') as file:
        info = pickle.load(file)

    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate Quality')
    parser.add_argument('--io-filepath', type=str, required=True)
    parser.add_argument('--alltimes-filepath', type=str, required=True)
    parser.add_argument('--image-filepath', type=str, required=True)

    stage = 'total'
    arguments = parser.parse_args()

    io = load_pickle(arguments.io_filepath)
    alltimes = load_pickle(arguments.alltimes_filepath)

    # i2t_i = list()
    # i2t_t = list()
    # for alltime in alltimes[stage]:
    #     assert len(io) == len(alltime)
    #     for index, ((iii, iii_ap, ooo), (batch_id, batch_time)) in enumerate(zip(io, alltime.items())):
    #         i2t_i.append(iii_ap)
    #         i2t_t.append(batch_time)

    i2t = [(iii_ap, list()) for iii, iii_ap, _ in io]
    # i2t = [(iii, list()) for iii, iii_ap, _ in io]
    for alltime in alltimes[stage]:
        assert len(io) == len(alltime)
        for index, ((iii, iii_ap, ooo), (batch_id, batch_time)) in enumerate(zip(io, alltime.items())):
            assert batch_id == index + 1
            # assert i2t[index][0] == iii
            assert i2t[index][0] == iii_ap
            i2t[index][1].append(batch_time)

    i2t = sorted(i2t, key=lambda x: x[0])

    txt_sizes = [l for l, _ in i2t]

    unique_txt_sizes = dict()
    for img_size, t in i2t:
        unique_txt_sizes[img_size] = unique_txt_sizes.get(img_size, list()) + t

    i2t = list()
    for unique_txt_size, unique_t in unique_txt_sizes.items():
        i2t.append((unique_txt_size, unique_t))
    i2t = sorted(i2t, key=lambda x: x[0])

    txt_sizes = [l for l, _ in i2t]

    cmap = plt.get_cmap('coolwarm')
    fig, axes = plt.subplots(1, 1, figsize=(10, 10))
    ax = axes
    ax.grid(True, color='gray', alpha=0.3, linewidth=0.2)


    # all_l = [l for l in txt_sizes]
    # print(min(t_std), max(t_std))
    t_std = [numpy.std(this_t, ddof=1) for _, this_t in i2t]
    t_avg = [numpy.average(this_t) for _, this_t in i2t]
    q25th = [numpy.quantile(this_t, 0.25) for _, this_t in i2t]
    q50th = [numpy.quantile(this_t, 0.50) for _, this_t in i2t]
    q75th = [numpy.quantile(this_t, 0.75) for _, this_t in i2t]
    q99th = [numpy.quantile(this_t, 0.99) for _, this_t in i2t]
    t_min = [min(this_t) for _, this_t in i2t]
    t_max = [max(this_t) for _, this_t in i2t]

    # t_inv = list()
    # for _, this_t in i2t:
    #     this_total_inv = 0
    #     for single_t in this_t:
    #         if single_t > 0.3:
    #             this_total_inv += 1
    #     t_inv.append(this_total_inv)

    # ax.scatter(txt_sizes, t_inv, color='red',     marker='o', alpha=0.6, label='Average',         )
    # ax.scatter(txt_sizes, t_avg, color='red',     marker='o', alpha=0.6, label='Average',         )
    # ax.scatter(txt_sizes, t_min, color='green',     marker='x', alpha=0.6, label='Min',         )
    # ax.scatter(txt_sizes, t_max, color='orange',     marker='^', alpha=0.6, label='Max',         )
    # ax.scatter(txt_sizes, q25th, color='orange',  marker='x', alpha=0.6, label='25th percentile', )
    # ax.scatter(txt_sizes, q50th, color='green',   marker='x', alpha=0.6, label='Median',          )
    # ax.scatter(txt_sizes, q75th, color='cyan',    marker='x', alpha=0.6, label='75th percentile', )
    # ax.fill_between(txt_sizes, q99th, t_max, color='green', alpha=0.3)
    # ax.fill_between(txt_sizes, q25th, q99th, color='red', alpha=0.7)
    # ax.fill_between(txt_sizes, t_min, q25th, color='blue', alpha=0.7)
    # ax.plot(txt_sizes, t_max, 'b^', markersize=6)

    # for index, length in enumerate(txt_sizes):
    #     # ax.plot([length, length], [t_avg[index]-t_std[index], t_avg[index]+t_std[index]], color='blue', alpha=0.7)
    #     # ax.plot([length, length], [q99th[index], t_max[index]], color='#ccff33', alpha=0.9)
    #     ax.plot([length, length], [q75th[index], q99th[index]], color='#9b2226', alpha=0.7)
    #     ax.plot([length, length], [q25th[index], q75th[index]], color='#ee9b00', alpha=0.5)
    #     ax.plot([length, length], [t_min[index], q25th[index]], color='#001219', alpha=0.3)
    # ac = "#384B70"
    # bc = "#8FD14F"
    # cc = "#B8001F"
    ac = '#A594F9'
    bc = '#001219'
    cc = '#da2e00'
    ax.vlines(txt_sizes, q75th, q99th, linewidth=5.5, color=cc, alpha=0.9, label='75% ~ 99%')
    ax.vlines(txt_sizes, q25th, q75th, linewidth=5.5, color=bc, alpha=0.9, zorder=3, label='25% ~ 75%')
    ax.vlines(txt_sizes, t_min, q25th, linewidth=5.5, color=ac, alpha=0.9, label='  0% ~ 25%')

    # for i in range(len(txt_sizes)-1):
    #     ax.plot([txt_sizes[i], txt_sizes[i+1]], [max(q99th)+0.1]*2, color=cmap(t_max[i] / max(t_max)), linewidth=4)

    # for i, _ in enumerate(t_avg):
    #     if t_avg[i] - t_avg[i-1] > 0.05:
    #         print(t_avg[i-1], t_avg[i], i)


    ax.legend(title='Percentile Ranges', loc='upper left', title_fontsize=30, fontsize=30)
    ax.tick_params(axis='x', labelsize=25)
    ax.tick_params(axis='y', labelsize=25)
    ax.set_xlabel('Tokens per Prompt', fontsize=30)
    ax.set_ylabel('Inference Time (w/ pre-/post-process)', fontsize=30)

    plt.tight_layout()
    fig.savefig(arguments.image_filepath, bbox_inches='tight')