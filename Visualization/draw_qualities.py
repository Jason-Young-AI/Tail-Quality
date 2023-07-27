import io
import sys
import time
import numpy
import pathlib
import argparse
import multiprocessing

from matplotlib import pyplot

from constant import quality_choices, dataset_choices, combine_choices, rm_outs_choices
from calculate_quality import combine_ImageNet_times, combine_MMLU_times, combine_COCO_times, calculate_acc, calculate_map, calculate_wf1


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


def multiprocess_calculate_quality(results, dataset_type, assets_path, threshold, times, quality_type, order):
    tic = time.perf_counter()
    calculate_quality = globals()['calculate_' + quality_type]
    original_stdout = sys.stdout
    sys.stdout = io.StringIO()
    quality = calculate_quality(results, dataset_type, assets_path, threshold, times)
    sys.stdout = original_stdout
    toc = time.perf_counter()
    print(f" ^ === DONE:{toc-tic:.6f}s ({multiprocessing.current_process().pid}) Order: {order} === ^ ")
    return quality, order


def draw_coco_qualities(imgs_save_dir, main_results, combined_times, rm_outs_type, recompute=True, qualities_path=None):
    _, run_number = combined_times.shape
    print(f"Checking outliers (mode: {rm_outs_type})...")

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

    print(f'Total Lower Outlier:{len(lower_outliers)}')
    print(f'Total Upper Outlier:{len(upper_outliers)}')

    min_lower_thr = numpy.min(lower_threshold)
    max_upper_thr = numpy.max(upper_threshold)

    percentile = numpy.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]) * 100
    special_thresholds = numpy.percentile(combined_times, percentile, method='lower')

    threshold_step = (max_upper_thr - min_lower_thr) / 10
    thresholds = numpy.arange(min_lower_thr, max_upper_thr + threshold_step, threshold_step)

    origin_quality = calculate_map(main_results, 'COCO')

    if recompute:
        qualities = calculate_coco_qualities(main_results, numpy.concatenate([thresholds, special_thresholds]), combined_times)
        if qualities_path is not None:
            qualities_path = pathlib.Path(qualities_path)
            print(f" + Saving qualities into \'{qualities_path}.npz\' ...")
            numpy.savez(qualities_path, qualities)
            print(f" - Saved.")
    else:
        assert qualities_path is not None, "Parameter \{qualities_path\} must be set if parameter \{recompute\} is False."
        qualities_path = pathlib.Path(qualities_path)
        assert qualities_path.is_file(), f"No Such Qualities Bin File: {qualities_path}"
        qualities = numpy.load(qualities_path)
        assert qualities.shape == (len(thresholds) + len(special_thresholds), run_number)

    fig, axes = pyplot.subplots(1, 1, figsize=(10, 10))
    ax = axes

    ogn_color = 'xkcd:crimson'
    max_color = 'xkcd:darkblue'
    min_color = 'xkcd:chartreuse'
    avg_color = 'xkcd:indigo'

    ax.scatter(thresholds[-1], origin_quality, label='No Time Limit', color=ogn_color, marker='*', s=10, zorder=3)
    ax.annotate(f'q = {origin_quality*100:.3f}', xy=(thresholds[-1], origin_quality),
            xytext=(-4, numpy.sign(origin_quality)*3), textcoords="offset points",
            horizontalalignment="right",
            verticalalignment="bottom" if origin_quality > 0 else "top")

    # Special Thresholds
    special_mins = numpy.min(qualities[-len(special_thresholds):, :], axis=-1)
    special_maxs = numpy.max(qualities[-len(special_thresholds):, :], axis=-1)
    ax.vlines(special_thresholds, special_mins, special_maxs, ls=':', color='c', label=r'All $\alpha$s at $\theta$')
    ax.scatter(special_thresholds, special_maxs, label=r'Max quality at threshold', marker='v', edgecolors='g', facecolors='1', s=10, zorder=4)
    ax.scatter(special_thresholds, special_mins, label=r'Min quality at threshold', marker='^', edgecolors='m', facecolors='1', s=10, zorder=4)

    # Thresholds
    avgs = numpy.average(qualities[:len(thresholds)], axis=-1)
    mins = numpy.min(qualities[:len(thresholds)], axis=-1)
    maxs = numpy.max(qualities[:len(thresholds)], axis=-1)
    stds = numpy.std(qualities[:len(thresholds)], axis=-1)
    ax.plot(thresholds, maxs, label='Maximum Quality(mAP)', color=max_color, linewidth=1.0)
    ax.plot(thresholds, mins, label='Minimum Quality(mAP)', color=min_color, linewidth=1.0)
    ax.plot(thresholds, avgs, label='Average Quality(mAP)', color=avg_color, linewidth=1.0)
    ax.fill_between(thresholds, numpy.minimum(avgs + stds, maxs), numpy.maximum(avgs - stds, mins), label='Standard Deviation', color=avg_color, alpha=0.2)

    ax.set_xlabel('Inference Time Thresholds (sec.)', fontsize=8)
    ax.set_ylabel(f'Quality (mAP)', fontsize=8)
    ax.legend()

    figpath = imgs_save_dir.joinpath(f'qualities.pdf')
    fig.savefig(figpath)
    print(f' - Fig Exported: {figpath}')


def calculate_coco_qualities(main_results, thresholds, combined_times):
    _, run_number = combined_times.shape
    qualities = list()

    # [Begin] Multi-Processing
    multiprocessing.set_start_method('spawn')
    for t_index, threshold in enumerate(thresholds):
        tic = time.perf_counter()
        print(f"Threshold({threshold}) {t_index+1}/{len(thresholds)} ...")
        pool = multiprocessing.Pool(8)
        subprocesses = list()
        print(f" v Calculating Quality (mAP) at Threshold({threshold}) total Run({run_number}) ...")
        for r_index in range(run_number):
            subprocesses.append(
                pool.apply_async(
                    multiprocess_calculate_quality,
                    args=(main_results, 'COCO', pathlib.Path('assets'), threshold, combined_times[:, r_index], 'map', r_index)
                )
            )
        pool.close()
        pool.join()
        toc = time.perf_counter()
        print(f" - Consume time: {toc - tic:.2f}s")

        quality_at_threshold = numpy.ndarray(run_number)
        for subprocess in subprocesses:
            quality, r_index = subprocess.get()
            quality_at_threshold[r_index] = quality[0]

        qualities.append(quality_at_threshold)
    qualities = numpy.array(qualities)
    # [End] Multi-Processing
    return qualities


def draw_COCO(extracted_data, combine_type, rm_outs_type, imgs_save_dir, recompute=True, qualities_path=None):
    main_results = extracted_data['main_results']
    #inference_times = extracted_data['other_results']['inference_times']
    #preprocess_times = extracted_data['other_results']['preprocess_times']
    #postprocess_times = extracted_data['other_results']['postprocess_times']

    combined_times = combine_COCO_times(extracted_data['other_results'], combine_type) # numpy.array()[instance_number * run_number]

    print(f'[Begin] Drawing ...')

    # print(f' v Drawing ...')
    # draw_coco_count_per_nop(imgs_save_dir, bis_to_count)
    # print(f' ^ Draw Count V.S. Number of Pixels Finished.\n')

    print(f' v Drawing ...')
    draw_coco_qualities(imgs_save_dir, main_results, combined_times, rm_outs_type, recompute=recompute, qualities_path=qualities_path)
    print(f' ^ Draw Specific Statistics Finished.\n')

    print(f'[End] All Finished.')


def draw_ImageNet(extracted_data, combine_type, rm_outs_type, imgs_save_dir, recompute=True, qualities_path=None):
    pass


def draw(extracted_data, dataset_type, combine_type, rm_outs_type, imgs_save_dir, recompute=True, qualities_path=None):
    assert dataset_type in dataset_choices, f"Wrong Type of Dataset: {dataset_type}"
    assert combine_type in combine_choices, f"Wrong Type of Combine: {combine_type}"
    assert rm_outs_type in rm_outs_choices, f"Wrong Type of Remove Outliers: {rm_outs_type}"

    draw_by_dst = globals()['draw_' + dataset_type]
    draw_by_dst(extracted_data, combine_type, rm_outs_type, imgs_save_dir, recompute=recompute, qualities_path=qualities_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Draw Figs for Datasets')

    parser.add_argument('-i', '--imgs-save-dir', type=str, required=True)

    parser.add_argument('-d', '--data-dir', type=str, default=None)
    parser.add_argument('-n', '--data-filename', type=str, default=None)
    parser.add_argument('-s', '--save-dir', type=str, default=None)
    parser.add_argument('-f', '--save-filename', type=str, default=None)
    parser.add_argument('-p', '--npz-path', type=str, default=None)
    parser.add_argument('-l', '--recompute', type=bool, default=True)
    parser.add_argument('-q', '--qualities-path', type=str, default=None)
    parser.add_argument('-t', '--dataset-type', type=str, default='ImageNet', choices=dataset_choices)
    parser.add_argument('-c', '--combine-type', type=str, default='i', choices=combine_choices)
    parser.add_argument('-r', '--rm-outs-type', type=str, default='gaussian', choices=rm_outs_choices)
    arguments = parser.parse_args()

    dataset_type = arguments.dataset_type
    assert dataset_type in dataset_choices, f"No Such Dataset Type: {dataset_type}"

    combine_type = arguments.combine_type
    assert combine_type in combine_choices, f"No Such Combine Type: {combine_type}"

    rm_outs_type = arguments.rm_outs_type
    assert rm_outs_type in rm_outs_choices, f"No Such Dataset Type: {rm_outs_type}"

    imgs_save_dir = pathlib.Path(arguments.imgs_save_dir)

    if not imgs_save_dir.is_dir():
        print(f"No Imgs Save Dir Exists: {imgs_save_dir}, now creating it.")
        imgs_save_dir.mkdir(parents=True, exist_ok=True)

    if arguments.data_dir is None:
        # Direct Load From NPZ
        if arguments.npz_path is None:
            raise AttributeError("At least one argument of \{--data-dir or --npz-path\} must be specified.")
        else:
            npz_path = pathlib.Path(arguments.npz_path)
            assert npz_path.is_file(), f"No Such NPZ File: {npz_path}"
            extracted_data = numpy.load(npz_path)
    else:
        # Load From Raw Data
        data_dir = pathlib.Path(arguments.data_dir)
        data_filename = arguments.data_filename
        assert data_dir.is_dir(), f"No Such Data Dir: {data_dir}"
        assert data_filename is not None, f"While using argument \'--data-dir\', one must specify \'--data-filename\'"

        from extract_data import extract_data
        extracted_data = extract_data(data_dir, data_filename, dataset_type)

        if arguments.save_dir is None:
            print(f"The extracted data will not be saved!\n")
            print(f"If one want to save the extracted data, please specify arguments \'--save-dir\' and \'--save-filename\'")
        else:
            save_dir = pathlib.Path(arguments.save_dir)
            save_filename = arguments.save_filename
            assert save_dir.is_dir(), f"No Such Save Dir: {save_dir}"
            assert save_filename is not None, f"While using argument \'--save-dir\', one must specify \'--save-filename\'"
            save_filepath = save_dir.joinpath(save_filename)
            print(f" + Saving data into \'{save_filepath}.npz\' ...")
            numpy.savez(save_filepath, **extracted_data)
            print(f" - Saved.")

    draw(extracted_data, dataset_type, combine_type, rm_outs_type, imgs_save_dir, arguments.recompute, arguments.qualities_path)