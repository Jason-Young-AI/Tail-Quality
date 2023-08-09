import io
import sys
import time
import numpy
import pathlib
import argparse
import multiprocessing

import matplotlib as mpl
from matplotlib import pyplot

from constant import quality_choices, dataset_choices, combine_choices, rm_outs_choices, quality_map
from calculate_quality import combine_times, calculate_acc, calculate_map, calculate_wf1


# def draw_coco_count_per_nop(imgs_save_dir, bis_to_count):
#     # Number of Pixels
#     fig, ax = pyplot.subplots(1, 1, figsize=(10, 10))

#     nop_to_bnc = dict()
#     for bis, count in bis_to_count:
#         nop = bis[0] * bis[1]
#         bnc = nop_to_bnc.get(nop, list())
#         bnc.append((bis, count))
#         nop_to_bnc[nop] = bnc

#     nop_to_bnc = sorted(nop_to_bnc.items(), key=lambda x: x[0])
#     nop_diff = list()
#     for i in range(1, len(nop_to_bnc)):
#         nop_diff.append(nop_to_bnc[i][0] - nop_to_bnc[i-1][0])
#     nop_diff = sorted(enumerate(nop_diff), key=lambda x: x[1])[::-1]
#     print(len(nop_diff))
#     print(nop_diff)
#     grain_size = 1000


def multiprocess_calculate_quality(results, quality_type, dataset_type, assets_path, threshold, times, order, verbose):
    tic = time.perf_counter()
    calculate_quality = globals()['calculate_' + quality_type]
    original_stdout = sys.stdout
    sys.stdout = io.StringIO()
    quality = calculate_quality(results, dataset_type, assets_path, threshold, times)
    sys.stdout = original_stdout
    toc = time.perf_counter()
    if verbose:
        print(f"   === DONE:{toc-tic:.6f}s ({multiprocessing.current_process().pid}) Order: {order} === ^ ")
    return quality, order

def draw_qualities(save_dir, main_results, combined_times, quality_type, dataset_type, rm_outs_type, sub_proc_num, interplt_num, recompute, draw_npz_path, verbose):
    _, run_number = combined_times.shape
    print(f" . Checking outliers (mode: {rm_outs_type})...")

    lower_threshold = numpy.min(combined_times, axis=-1, keepdims=True)
    upper_threshold = numpy.max(combined_times, axis=-1, keepdims=True)

    if rm_outs_type == 'quantile':
        quantile_1 = numpy.quantile(combined_times, 0.25, axis=-1, keepdims=True)
        quantile_3 = numpy.quantile(combined_times, 0.75, axis=-1, keepdims=True)
        iqr = quantile_3 - quantile_1
        lower_threshold = numpy.maximum(quantile_1 - 1.5 * iqr, lower_threshold)
        upper_threshold = numpy.minimum(quantile_3 + 1.5 * iqr, upper_threshold)

    if rm_outs_type == 'gaussian':
        mean, std = numpy.mean(combined_times, axis=-1, keepdims=True), numpy.std(combined_times, axis=-1, keepdims=True)
        lower_threshold = numpy.maximum(mean - 3 * std, lower_threshold)
        upper_threshold = numpy.minimum(mean + 3 * std, upper_threshold)

    lower_outliers = combined_times[combined_times < lower_threshold]
    upper_outliers = combined_times[upper_threshold < combined_times]

    print(f' . Total Lower Outlier:{len(lower_outliers)}')
    print(f' . Total Upper Outlier:{len(upper_outliers)}')

    min_lower_thr = numpy.min(lower_threshold)
    max_upper_thr = numpy.max(upper_threshold)

    percentile = numpy.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]) * 100
    special_thresholds = numpy.percentile(combined_times, percentile, method='lower')

    threshold_step = (max_upper_thr - min_lower_thr) / interplt_num
    thresholds = numpy.arange(min_lower_thr, max_upper_thr + threshold_step, threshold_step)

    all_thresholds = numpy.concatenate([thresholds, special_thresholds])

    if not recompute:
        draw_data = numpy.load(draw_npz_path)
        qualities = draw_data['qualities']
        dd_thresholds = draw_data['thresholds']
        dd_special_thresholds = draw_data['special_thresholds']
        dd_all_thresholds = numpy.concatenate([dd_thresholds, dd_special_thresholds])
        if len(dd_thresholds) != len(thresholds) or len(dd_special_thresholds) != len(special_thresholds) or numpy.sum(all_thresholds != dd_all_thresholds) != 0:
            print(f" . Specified Draw Data NPZ file({draw_npz_path}) is invalid, qualities and thresholds will be recomputed!")
            recompute = True
        else:
            assert qualities.shape == (len(all_thresholds), run_number), "Invalid Draw Data NPZ file."
            print(f" . Qualities and Thresholds are loaded from Draw Data NPZ file({draw_npz_path})!")

    if recompute:
        print(" . Recomputing qualities and thresholds ...")
        qualities = calculate_qualities(main_results, all_thresholds, combined_times, quality_type, dataset_type, sub_proc_num, verbose)
        if draw_npz_path is not None:
            draw_data = dict(
                qualities = qualities,
                thresholds = thresholds,
                special_thresholds = special_thresholds
            )
            print(f" + Saving qualities into \'{draw_npz_path}' ...")
            numpy.savez(draw_npz_path, **draw_data)
            print(f" - Saved.")

    calculate_quality = globals()['calculate_' + quality_type]
    origin_quality = calculate_quality(main_results, dataset_type)

    if quality_type == 'map':
        origin_quality = origin_quality[0]

    if quality_type == 'acc':
        origin_quality = origin_quality[1]

    mpl.rcParams['font.size'] = 14
    mpl.rcParams['axes.labelsize'] = 16
    mpl.rcParams['xtick.labelsize'] = 14
    mpl.rcParams['ytick.labelsize'] = 14
    mpl.rcParams['legend.fontsize'] = 14

    fig, axes = pyplot.subplots(1, 1, figsize=(10, 10))
    ax = axes

    cmap = pyplot.get_cmap('tab20')
    #colors = [cmap(i) for i in numpy.linspace(0, 1, 3)]

    ogn_color = cmap(6)
    max_color = cmap(10)
    maxs_color = cmap(12)
    min_color = cmap(14)
    mins_color = cmap(16)
    avg_color = cmap(0)
    vln_color = cmap(15)

    ax.scatter(thresholds[-1]*1000, origin_quality, label='No Time Limit', color=ogn_color, marker='*', s=15, zorder=3)
    ax.annotate(f'{origin_quality*100:.3f}', xy=(thresholds[-1]*1000, origin_quality),
            xytext=(-4, numpy.sign(origin_quality)*3), textcoords="offset points",
            horizontalalignment="right",
            verticalalignment="bottom" if origin_quality > 0 else "top")

    # Special Thresholds
    special_mins = numpy.min(qualities[-len(special_thresholds):, :], axis=-1)
    special_maxs = numpy.max(qualities[-len(special_thresholds):, :], axis=-1)
    ax.vlines(special_thresholds*1000, special_mins, special_maxs, ls=':', color=vln_color, zorder=4)#, label=r'All $\alpha$s at $\theta$')
    ax.scatter(special_thresholds*1000, special_maxs, marker='v', edgecolors=maxs_color, facecolors='1', s=15, zorder=4)
    ax.scatter(special_thresholds*1000, special_mins, marker='^', edgecolors=mins_color, facecolors='1', s=15, zorder=4)
    for x, ymin, ymax in zip(special_thresholds*1000, special_mins, special_maxs):
        ymid = (ymin+ymax) / 2
        # ax.annotate(f'ltc = {x:.5f}', xy=(x, ymin),
        #         xytext=(+0, -5 * np.sign(ymid)*3), textcoords="offset points",
        #         horizontalalignment="center",
        #         verticalalignment="top" if ymid > 0 else "bottom")
        ax.annotate(f'{ymin*100:.2f} ({x:.2f}ms)', xy=(x, ymin),
                xytext=(+8, 0), textcoords="offset points",
                horizontalalignment="left",
                verticalalignment="top" if ymin > 0 else "bottom")
        ax.annotate(fr'{ymax*100:.2f}', xy=(x, ymax),
                xytext=(-4, numpy.sign(ymax)*3), textcoords="offset points",
                horizontalalignment="right",
                verticalalignment="bottom" if ymax > 0 else "top")

    # Thresholds
    origin_indices = list()
    sorted_all_thresholds = list()
    for origin_index, threshold in sorted(enumerate(all_thresholds), key=lambda x: x[1]):
        origin_indices.append(origin_index)
        sorted_all_thresholds.append(threshold)

    sorted_all_thresholds = numpy.array(sorted_all_thresholds)
    avgs = numpy.average(qualities, axis=-1)[origin_indices]
    mins = numpy.min(qualities, axis=-1)[origin_indices]
    maxs = numpy.max(qualities, axis=-1)[origin_indices]
    stds = numpy.std(qualities, axis=-1)[origin_indices]
    ax.plot(sorted_all_thresholds*1000, maxs, label=f'Maximum {quality_map[quality_type]}', color=max_color, linewidth=2.0)
    ax.plot(sorted_all_thresholds*1000, mins, label=f'Minimum {quality_map[quality_type]}', color=min_color, linewidth=2.0)
    ax.plot(sorted_all_thresholds*1000, avgs, label=f'Average {quality_map[quality_type]}', color=avg_color, linewidth=2.0)
    ax.fill_between(sorted_all_thresholds*1000, numpy.minimum(avgs + stds, maxs), numpy.maximum(avgs - stds, mins), color=avg_color, alpha=0.2)

    ax.set_xlabel('Inference Time Thresholds (Milliseconds)')
    ax.set_ylabel(f'Inference Quality ({quality_map[quality_type]})')
    ax.legend()

    figpath = save_dir.joinpath(f'qualities.pdf')
    fig.savefig(figpath)
    print(f' - Fig Exported: {figpath}')


def calculate_qualities(main_results, thresholds, combined_times, quality_type, dataset_type, sub_proc_num, verbose):
    _, run_number = combined_times.shape
    qualities = list()

    # [Begin] Multi-Processing
    multiprocessing.set_start_method('spawn')
    for t_index, threshold in enumerate(thresholds):
        tic = time.perf_counter()
        print(f"Threshold({threshold}) {t_index+1}/{len(thresholds)} ...")
        pool = multiprocessing.Pool(sub_proc_num)
        subprocesses = list()
        print(f" v Calculating Quality ({quality_map[quality_type]}) at Threshold({threshold}) total Run({run_number}) ...")
        for r_index in range(run_number):
            subprocesses.append(
                pool.apply_async(
                    multiprocess_calculate_quality,
                    args=(main_results, quality_type, dataset_type, pathlib.Path('assets'), threshold, combined_times[:, r_index], r_index, verbose)
                )
            )
        pool.close()
        pool.join()
        toc = time.perf_counter()
        print(f" ^ Consume time: {toc - tic:.2f}s")

        quality_at_threshold = numpy.ndarray(run_number)
        for subprocess in subprocesses:
            quality, r_index = subprocess.get()
            if quality_type == 'map':
                quality = quality[0]

            if quality_type == 'acc':
                quality = quality[1]

            quality_at_threshold[r_index] = quality

        qualities.append(quality_at_threshold)
    qualities = numpy.array(qualities)
    # [End] Multi-Processing
    return qualities


def draw(extracted_data, quality_type, dataset_type, combine_type, rm_outs_type, save_dir, sub_proc_num, interplt_num, recompute, qlts_npz_path, verbose):
    assert dataset_type in dataset_choices, f"Wrong Type of Dataset: {dataset_type}"
    assert combine_type in combine_choices, f"Wrong Type of Combine: {combine_type}"
    assert rm_outs_type in rm_outs_choices, f"Wrong Type of Remove Outliers: {rm_outs_type}"

    main_results = extracted_data['main_results']
    #inference_times = extracted_data['other_results']['inference_times']
    #preprocess_times = extracted_data['other_results']['preprocess_times']
    #postprocess_times = extracted_data['other_results']['postprocess_times']

    combined_times = combine_times(extracted_data['other_results'], combine_type)[:, :20] # numpy.array()[instance_number * run_number]

    print(f'[Begin] Drawing ...')

    # print(f' v Drawing ...')
    # draw_coco_count_per_nop(imgs_save_dir, bis_to_count)
    # print(f' ^ Draw Count V.S. Number of Pixels Finished.\n')

    print(f' v Drawing ...')
    draw_qualities(save_dir, main_results, combined_times, quality_type, dataset_type, rm_outs_type, sub_proc_num, interplt_num, recompute, qlts_npz_path, verbose)
    print(f' ^ Draw COCO Qualities Finished.\n')

    print(f'[End] All Finished.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Draw Figs for Datasets')

    parser.add_argument('-s', '--save-dir', type=str, required=True)

    parser.add_argument('-d', '--data-dir', type=str, required=True)
    parser.add_argument('-n', '--data-filename', type=str, required=True)
    #parser.add_argument('-p', '--data-npz-path', type=str, default=None)

    parser.add_argument('-l', '--recompute', action='store_true', default=False)
    parser.add_argument('-u', '--draw-npz-path', type=str, default=None)
    parser.add_argument('-m', '--sub-proc-num', type=int, default=8)
    parser.add_argument('-w', '--interplt-num', type=int, default=100)

    parser.add_argument('-q', '--quality-type', type=str, default='acc', choices=quality_choices)
    parser.add_argument('-t', '--dataset-type', type=str, default='ImageNet', choices=dataset_choices)
    parser.add_argument('-c', '--combine-type', type=str, default='i', choices=combine_choices)
    parser.add_argument('-r', '--rm-outs-type', type=str, default='gaussian', choices=rm_outs_choices)

    parser.add_argument('-v', '--verbose', action='store_true')
    arguments = parser.parse_args()

    quality_type = arguments.quality_type
    dataset_type = arguments.dataset_type
    combine_type = arguments.combine_type
    rm_outs_type = arguments.rm_outs_type

    save_dir = pathlib.Path(arguments.save_dir)

    if not save_dir.is_dir():
        print(f"No Imgs Save Dir Exists: {save_dir}, now creating it.")
        save_dir.mkdir(parents=True, exist_ok=True)

    #if arguments.data_dir is None:
    #    # Direct Load From NPZ
    #    if arguments.data_npz_path is None:
    #        raise AttributeError("At least one argument of \{--data-dir or --data-npz-path\} must be specified.")
    #    else:
    #        data_npz_path = pathlib.Path(arguments.data_npz_path)
    #        data_npz_path = data_npz_path.with_suffix('.npz')
    #        assert data_npz_path.is_file(), f"No Such Data NPZ File: {data_npz_path}"
    #        extracted_data = numpy.load(data_npz_path, allow_pickle=True)
    #        print(f" . extracted_data loaded from Data NPZ File({data_npz_path})")
    #else:
    # Load From Raw Data
    data_dir = pathlib.Path(arguments.data_dir)
    data_filename = arguments.data_filename
    assert data_dir.is_dir(), f"No Such Data Dir: {data_dir}"
    assert data_filename is not None, f"While using argument \'--data-dir\', one must specify \'--data-filename\'"

    from extract_data import extract_data
    print(f" . extracted_data loading from Raw Data Dir({data_dir}) with filename({data_filename})")
    extracted_data = extract_data(data_dir, data_filename, dataset_type)

    if not arguments.recompute and arguments.draw_npz_path is None:
        raise AttributeError("Argument \{--draw-npz-path\} must be specified if argument \{--recompute\} is not set.")
    else:
        recompute = arguments.recompute
        if arguments.draw_npz_path is None:
            draw_npz_path = None
        else:
            draw_npz_path = pathlib.Path(arguments.draw_npz_path)
            draw_npz_path = draw_npz_path.with_suffix('.npz')
            if not recompute:
                assert draw_npz_path.is_file(), f"No Such Qualities NPZ File: {draw_npz_path}"

    sub_proc_num = arguments.sub_proc_num
    assert sub_proc_num > 0, f"At least 1 subprocess, instead --sub-proc-num = {sub_proc_num}"

    interplt_num = arguments.interplt_num
    assert interplt_num > 9, f"At least 10 interpoint to plot, instead --interplot-num = {interplt_num}"

    draw(extracted_data, quality_type, dataset_type, combine_type, rm_outs_type, save_dir, sub_proc_num, interplt_num, recompute, draw_npz_path, arguments.verbose)