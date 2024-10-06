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

    i2t = list()
    for iii, iii_ap, _ in io:
        i2t.append((iii_ap, list()))
        # if iii_ap[0] < iii_ap[1]:
        #     i2t.append((iii_ap, list()))
        # else:
        #     i2t.append(((iii_ap[1], iii_ap[0]), list()))

    for alltime in alltimes[stage]:
        assert len(io) == len(alltime)
        for index, ((iii, iii_ap, ooo), (batch_id, batch_time)) in enumerate(zip(io, alltime.items())):
            assert batch_id == index + 1
            # assert i2t[index][0] == iii_ap
            i2t[index][1].append(batch_time)

    i2t = sorted(i2t, key=lambda x: (x[0][0], x[0][1]))

    img_sizes = [(h, w) for (h, w), _ in i2t]

    unique_img_sizes = dict()
    for img_size, t in i2t:
        unique_img_sizes[img_size] = unique_img_sizes.get(img_size, list()) + t

    i2t = list()
    for unique_img_size, unique_t in unique_img_sizes.items():
        i2t.append((unique_img_size, unique_t))
    i2t = sorted(i2t, key=lambda x: (x[0][0], x[0][1]))

    img_sizes = [(h, w) for (h, w), _ in i2t]

    fig, axes = plt.subplots(1, 1, figsize=(10, 10))
    ax = axes
    ax.grid(True, color='gray', alpha=0.6, linewidth=0.2)


    # all_h = [h for (h, w) in img_sizes]
    # all_w = [w for (h, w) in img_sizes]
    # t_std = [numpy.std(this_t, ddof=1)*10000 for _, this_t in i2t]
    # print(min(t_std), max(t_std))
    # t_avg = [numpy.average(this_t) for _, this_t in i2t]

    # scatter = ax.scatter(all_h, all_w, s=t_std, c=t_avg, cmap='coolwarm', alpha=0.6)
    # cbar = fig.colorbar(scatter, ax=ax)

    llr_i2t = [((h, w), t) for (h, w), t in i2t if h <= w]
    rll_i2t = [((w, h), t) for (h, w), t in i2t if h >  w]

    llr_all_s = [s[0]*s[1] for s, t in llr_i2t]
    llr_all_t = [numpy.average(t) for s, t in llr_i2t]
    # llr_all_00th = [numpy.quantile(t, 0.00) for s, t in llr_i2t]
    # llr_all_25th = [numpy.quantile(t, 0.25) for s, t in llr_i2t]
    # llr_all_75th = [numpy.quantile(t, 0.75) for s, t in llr_i2t]
    # llr_all_99th = [numpy.quantile(t, 0.99) for s, t in llr_i2t]
    # llr_all_100th = [max(t) for s, t in llr_i2t]
    rll_all_s = [s[0]*s[1] for s, t in rll_i2t]
    rll_all_t = [numpy.average(t) for s, t in rll_i2t]
    # rll_all_00th = [numpy.quantile(t, 0.00) for s, t in rll_i2t]
    # rll_all_25th = [numpy.quantile(t, 0.25) for s, t in rll_i2t]
    # rll_all_75th = [numpy.quantile(t, 0.75) for s, t in rll_i2t]
    # rll_all_99th = [numpy.quantile(t, 0.99) for s, t in rll_i2t]

    # ax.vlines(llr_all_s, llr_all_00th, llr_all_25th, color='orange', alpha=0.6)
    # ax.vlines(llr_all_s, llr_all_75th, llr_all_99th, color='purple', alpha=0.6)
    # ax.vlines(llr_all_s, llr_all_99th, llr_all_100th, color='black', alpha=0.6)

    # ax.vlines(rll_all_s, rll_all_00th, rll_all_25th, color='cyan', alpha=0.6)
    # ax.vlines(rll_all_s, rll_all_75th, rll_all_99th, color='black', alpha=0.6)

    all_s = [s[0]*s[1] for s, t in i2t]
    # all_t = [numpy.average(t) for s, t in i2t]
    all_00th = [numpy.quantile(t, 0.00) for s, t in i2t]
    all_25th = [numpy.quantile(t, 0.25) for s, t in i2t]
    all_75th = [numpy.quantile(t, 0.75) for s, t in i2t]
    all_99th = [numpy.quantile(t, 0.99) for s, t in i2t]

    ac = '#A594F9'
    # bc = '#001219'
    cc = '#da2e00'
    # sac = "#f9ea9a"
    # sac = "#ffc600"
    sac = "#a7c957"
    sbc = "#001219"
    vl2 = ax.vlines(all_s, all_75th, all_99th, linewidth=3.8, color=cc, alpha=0.6, label='75% ~ 99%')
    vl1 = ax.vlines(all_s, all_00th, all_25th, linewidth=3.8, color=ac, alpha=0.6, label='  0% ~ 25%')
    # sc1 = ax.scatter(llr_all_s, llr_all_t, zorder=3, facecolors=sbc, marker='<', edgecolors=sbc, s=150, alpha=0.9, label='Width < Height')
    # sc2 = ax.scatter(rll_all_s, rll_all_t, zorder=3, facecolors=sac, marker='>', edgecolors=sac, s=150, alpha=0.9, label='Width > Height')
    sc1 = ax.scatter(llr_all_s, llr_all_t, zorder=3, facecolors='none', edgecolors=sbc, marker='<', s=150, alpha=0.9, label='Width < Height', linewidths=3)
    sc2 = ax.scatter(rll_all_s, rll_all_t, zorder=3, facecolors='none', edgecolors=sac, marker='>', s=150, alpha=0.9, label='Width > Height', linewidths=3)

    # ax.scatter(all_s, all_t, color='red', alpha=0.6)

    lg1 = ax.legend(handles=[vl2, vl1], title='Percentile Ranges', loc='upper left', fontsize=23, title_fontproperties={'weight':'bold', 'size': 26})
    lg2 = ax.legend(handles=[sc1, sc2], title='Averages', loc='lower right', fontsize=23, title_fontproperties={'weight':'bold', 'size': 26})
    # z2 = numpy.polyfit(llr_all_s, llr_all_t, 2)  # 线性拟合
    # p2 = numpy.poly1d(z2)
    # ax.plot(llr_all_s, p2(llr_all_s), "r--", label="Trend (Width > Length)")


    ax.ticklabel_format(style='scientific', axis='x', scilimits=(0, 0))
    ax.xaxis.get_offset_text().set_fontsize(25)

    ax.add_artist(lg1)
    ax.add_artist(lg2)
    ax.tick_params(axis='x', labelsize=25)
    ax.tick_params(axis='y', labelsize=25)
    ax.set_xlabel('Image Pixel Count', fontsize=30)
    ax.set_ylabel('Inference Time (w/ pre-/post-process)', fontsize=30)

    plt.tight_layout()
    fig.savefig(arguments.image_filepath, bbox_inches='tight')