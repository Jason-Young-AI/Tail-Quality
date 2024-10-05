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

    i2t = [(iii, list()) for iii, _ in io]
    for alltime in alltimes[stage]:
        assert len(io) == len(alltime)
        for index, ((iii, ooo), (batch_id, batch_time)) in enumerate(zip(io, alltime.items())):
            assert batch_id == index + 1
            assert i2t[index][0] == iii
            i2t[index][1].append(batch_time)

    i2t = sorted(i2t, key=lambda x: (x[0][0], x[0][1]))

    img_sizes = [(h, w) for (h, w), _ in i2t]

    fig, axes = plt.subplots(1, 1, figsize=(10, 10))
    ax = axes


    all_h = [h for (h, w) in img_sizes]
    all_w = [w for (h, w) in img_sizes]
    t_std = [numpy.std(this_t, ddof=1)*1000 for _, this_t in i2t]
    print(min(t_std), max(t_std))
    t_avg = [numpy.average(this_t) for _, this_t in i2t]

    ax.scatter(all_h, all_w, s=t_std, c=t_avg, cmap='viridis', alpha=0.6)

    ax.set_xlabel('Image Size')
    ax.set_ylabel('Frequencies')

    fig.savefig(arguments.image_filepath)