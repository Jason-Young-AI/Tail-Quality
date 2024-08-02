import json
import time
import numpy
import pickle
import pathlib
import multiprocessing

import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import Dict, List


def draw_tail_quality_acc(thresholds: List[numpy.ndarray], save_dir: pathlib.Path, topk, time_type):
    interplt_num = len(thresholds)-1
    all_qualities_path = pathlib.Path(save_dir.joinpath(f'{topk}_acc_{time_type}_points-{interplt_num}.json'))

    assert all_qualities_path.exists() , f'file {all_qualities_path} not found, please generate all_qualities first'
    with open(all_qualities_path, 'r') as f:
        all_qualities = json.load(f)

    mpl.rcParams['font.size'] = 14
    mpl.rcParams['axes.labelsize'] = 16
    mpl.rcParams['xtick.labelsize'] = 14
    mpl.rcParams['ytick.labelsize'] = 14
    mpl.rcParams['legend.fontsize'] = 14
    cmap = plt.get_cmap('tab20')
    ogn_color = cmap(6)
    max_color = cmap(10)
    maxs_color = cmap(12)
    min_color = cmap(14)
    mins_color = cmap(16)
    avg_color = cmap(0)
    vln_color = cmap(15)
    fig, axes = plt.subplots(1, 1, figsize=(10, 10))
    ax = axes
    origin_quality = max(all_qualities[-1].values())

    ax.scatter(thresholds[-1]*1000, origin_quality, label='No Time Limit', color=ogn_color, marker='*', s=15, zorder=3)
    ax.annotate(f'{origin_quality:.2f}', xy=(thresholds[-1]*1000, origin_quality),
            xytext=(-4, numpy.sign(origin_quality)*3), textcoords="offset points",
            horizontalalignment="right",
            verticalalignment="bottom" if origin_quality > 0 else "top")

    maxs = numpy.array([max(list(d.values())) for d in all_qualities])
    mins = numpy.array([min(list(d.values())) for d in all_qualities])
    avgs = numpy.array([sum(list(d.values())) / len(d.values()) for d in all_qualities])
    stds = numpy.array([numpy.std(list(d.values())) for d in all_qualities])

    ax.plot(thresholds*1000, maxs, label=f'Maximum', color=max_color, linewidth=2.0)
    ax.plot(thresholds*1000, mins, label=f'Minimum', color=min_color, linewidth=2.0)
    ax.plot(thresholds*1000, avgs, label=f'Average', color=avg_color, linewidth=2.0)
    ax.fill_between(thresholds*1000, numpy.minimum(avgs + stds, maxs), numpy.maximum(avgs - stds, mins), color=avg_color, alpha=0.2)
    
    ax.set_xlabel('Inference Time Thresholds (Milliseconds)')
    ax.set_ylabel(f'Inference Quality (Acc)')
    ax.legend(loc='lower right')
    plt.tight_layout()

    save_path = save_dir.joinpath(f'{topk}_acc_{time_type}_points-{interplt_num}') 
    plt.savefig(f'{save_path}.jpg', format='jpg', dpi=300)
    plt.savefig(f'{save_path}.pdf', format='pdf')
    print(f' - Fig saved')
    


def multiprocess_calculate_quality_acc(time_results, origin_quality, threshold, t_index, metric_type):
    if metric_type == 'acc':
        quality_at_threshold = dict()
        for run_index, times_result in enumerate(time_results, start=1): 
            batch_size = len(origin_quality)
            acc_tq = 0
            for batch_index, batch_time in times_result.items():
                if batch_time > threshold:
                    acc_tq += 0
                else:
                    acc_tq += origin_quality[str(batch_index)] 
            acc_tq = round((acc_tq / batch_size), 6)
            quality_at_threshold[run_index] = acc_tq
        return t_index, quality_at_threshold
    

def calculate_tail_quality_acc(time_results: List[Dict[str,float]], thresholds: List[numpy.ndarray], origin_quality: Dict[str,float], save_dir: pathlib.Path, topk, time_type, worker_number: int):
    interplt_num = len(thresholds)-1
    save_path = save_dir.joinpath(f'{topk}_acc_{time_type}_points-{interplt_num}.json') 
    all_batch_qualities = list()
    subprocesses = list()
    print('Now processing the data')
    
    pool = multiprocessing.Pool(worker_number)
    tqdm_bar1 = tqdm(total=len(thresholds), position=0)
    for t_index, threshold in enumerate(thresholds):
        tqdm_bar1.set_description(f"Threshold({threshold}) {t_index}/{len(thresholds)} ...")
        subprocesses.append(
            pool.apply_async(
                multiprocess_calculate_quality_acc,
                args=(time_results, origin_quality, threshold, t_index, 'acc')
            )
        )
        tqdm_bar1.update(1)
    pool.close()
    pool.join()
    print('Postprocessing the data')
    raw_list = list()
    for subprocess in subprocesses:
        t_index, quality_at_threshold = subprocess.get()
        raw_list.append((t_index, quality_at_threshold))
    raw_list.sort(key=lambda x: x[0])

    for (t_index, quality_at_threshold) in raw_list:
        all_batch_qualities.append(quality_at_threshold)
    print('Saving the results')
    with open(save_path, 'w') as f:
        json.dump(all_batch_qualities, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate Tail Quality and then visualize it")
    parser.add_argument('--results-basepath', type=str, required=True)
    parser.add_argument('--metric-type', type=str, required=True)
    parser.add_argument('--time-type', type=str, required=True)
    parser.add_argument('--worker-number', type=int, required=True)
    parser.add_argument('--interplt-num', type=int, required=True)
    args = parser.parse_args()

    results_basepath = pathlib.Path(args.results_basepath)
    metric_type = args.metric_type
    time_type = args.time_type
    worker_number = args.worker_number
    interplt_num = args.interplt_num

    result_path = results_basepath.joinpath('All_Times.pickle')
    rjsds_path = results_basepath.joinpath('All_rJSDs.pickle')
    fit_distribution_dir = results_basepath.joinpath('All_PDFs')
    origin_quality_path = results_basepath.joinpath('Origin_Quality.json')

    all_inference_times = list()
    all_total_times = list() 
    assert result_path.exists(), f'"{result_path}" not found'
    with open(result_path, 'rb') as f:
        results = pickle.load(f) 
        tmp_results = results.copy()
        for inference_times in tmp_results['inference']:
            for inference_time in inference_times.values():
                all_inference_times.append(inference_time)
        for total_times in tmp_results['total']:
            for total_time in total_times.values():
                all_total_times.append(total_time)
        del results
    # results : Dict = {
    #      'inference' : [
    #               {
    #                  'batch_id1' : time1,
    #                  'batch_id2' : time2,                 
    #                   ......
    #               }
    #               .....
    #       ]               
    #       'total' :......
    # }    
    # 

    if metric_type == 'acc':
        assert origin_quality_path.exists(), f'"{origin_quality_path}" not found'
        with open(origin_quality_path, 'r') as f:
            origin_quality = json.load(f)
            tmp_origin_quality = origin_quality.copy()
            top1_acc_origin_qualities = tmp_origin_quality['top1_acc'] 
            top5_acc_origin_qualities = tmp_origin_quality['top5_acc'] 
            del origin_quality
    # origin_quality : Dict = {
    #      'top1_acc' : {
    #               'batch_id' : quality,
    #                ......
    #           }
    #      'top5_acc' : {
    #                ......          
    #           }
    # }  
    
        all_inference_times = numpy.array(all_inference_times)  
        max_upper_inference_thr = numpy.max(all_inference_times)
        min_upper_inference_thr = numpy.min(all_inference_times)
        inference_threshold_step = (max_upper_inference_thr - min_upper_inference_thr) / interplt_num
        inference_thresholds = numpy.arange(min_upper_inference_thr, max_upper_inference_thr + inference_threshold_step, inference_threshold_step) 

        all_total_times = numpy.array(all_total_times)  
        max_upper_total_thr = numpy.max(all_total_times)
        min_upper_total_thr = numpy.min(all_total_times)
        total_threshold_step = (max_upper_total_thr - min_upper_total_thr) / interplt_num
        total_thresholds = numpy.arange(min_upper_total_thr, max_upper_total_thr + total_threshold_step, total_threshold_step)  
        
        # for debug
        print(max_upper_inference_thr)
        print(min_upper_inference_thr)
        print(numpy.max(inference_thresholds))
        print(numpy.min(inference_thresholds))
        print(type(inference_thresholds))
        print(inference_threshold_step)
        print(inference_thresholds.shape) 

        multiprocessing.set_start_method('spawn')

        if metric_type == 'acc':
            if time_type == 'inference':
                calculate_tail_quality_acc(tmp_results['inference'], inference_thresholds, top1_acc_origin_qualities, results_basepath, 'top1', time_type, worker_number)
                draw_tail_quality_acc(inference_thresholds, results_basepath, 'top1', time_type)
                calculate_tail_quality_acc(tmp_results['inference'], inference_thresholds, top5_acc_origin_qualities, results_basepath, 'top5', time_type, worker_number)
                draw_tail_quality_acc(inference_thresholds, results_basepath, 'top5', time_type)
            elif time_type == 'total':
                calculate_tail_quality_acc(tmp_results['total'], total_thresholds, top1_acc_origin_qualities, results_basepath, 'top1', time_type, worker_number)
                draw_tail_quality_acc(total_thresholds, results_basepath, 'top1', time_type)
                calculate_tail_quality_acc(tmp_results['total'], total_thresholds, top5_acc_origin_qualities, results_basepath, 'top5', time_type, worker_number)
                draw_tail_quality_acc(total_thresholds, results_basepath, 'top5', time_type)

    

    
