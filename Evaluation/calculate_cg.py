import sys
import numpy
import pathlib
import argparse

from .metrics              import Task
from .metrics.detr         import DETR
from .metrics.vicuna       import Vicuna
from .metrics.light_gcn    import LightGCN
from .metrics.hybrid_nets  import HybridNets
from .metrics.mobile_net   import MobileNet
from .metrics.emotion_flow import EmotionFlow

from .metrics.mobile_net_faster   import MobileNetFaster
from .metrics.light_gcn_faster    import LightGCNFaster

from .utils.io import save_pickle, load_pickle
from .utils.times import remove_outliers, get_outlier_flags
from .tail_quality import tail_quality
from .calculate_tl import get_tail_latency


tasks: dict[str, Task] = dict(
    detr         = DETR,
    vicuna       = Vicuna,
    light_gcn    = LightGCNFaster,
    hybrid_nets  = HybridNets,
    mobile_net   = MobileNetFaster,
    emotion_flow = EmotionFlow,
)


def swipe_all_times(rjsd_stop: float, all_rjsds: dict[str, list[int]], all_times: dict[str, list[dict[int, float]]], fit_run_number: int, window_size: int, warm_run: int) -> tuple[list[dict[int, float]], list[dict[int, float]]]:
    fit_distribution_number = 0

    swiped_inference_all_times = list()
    swiped_total_all_times = list()

    inference_all_times = all_times['inference']
    total_all_times = all_times['total']
    assert len(inference_all_times) == len(total_all_times)

    inference_all_rjsds = all_rjsds['inference']
    total_all_rjsds = all_rjsds['total']
    assert len(inference_all_rjsds) == len(total_all_rjsds)

    min_inference_all_rjsds = min(inference_all_rjsds)
    min_total_all_rjsds = min(total_all_rjsds)

    if min_inference_all_rjsds <= rjsd_stop:
        inference_yes = True
        inference_rjsd_converge = False
    else:
        inference_yes = False
        inference_rjsd_converge = True

    if min_total_all_rjsds <= rjsd_stop:
        total_yes = True
        total_rjsd_converge = False
    else:
        total_yes = False
        total_rjsd_converge = True

    for already_run, (inference_times, total_times) in enumerate(zip(inference_all_times, total_all_times), start=1):
        if inference_rjsd_converge and total_rjsd_converge:
            break

        if not inference_rjsd_converge:
            swiped_inference_all_times.append(inference_times)
        if not total_rjsd_converge:
            swiped_total_all_times.append(total_times)

        if already_run > warm_run and (already_run - warm_run) % fit_run_number == 0:
            inference_rjsd = all_rjsds['inference'][fit_distribution_number]
            total_rjsd = all_rjsds['total'][fit_distribution_number]
            if fit_distribution_number % window_size == 0 and fit_distribution_number != 0:
                if not inference_rjsd_converge and inference_rjsd <= rjsd_stop:
                    inference_rjsd_converge = True
                if not total_rjsd_converge and total_rjsd <= rjsd_stop:
                    total_rjsd_converge = True

            fit_distribution_number += 1
            if fit_distribution_number >= len(all_rjsds['inference']):
                break

    return inference_yes, swiped_inference_all_times, total_yes, swiped_total_all_times



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate Tail Quality")
    parser.add_argument('-i', '--ip-dirpath', type=str, required=True)
    parser.add_argument('-o', '--op-dirpath', type=str, required=True)

    parser.add_argument('--rjsd-stops', type=float, nargs='*')

    # Must Be The Same As Experiment Settings
    parser.add_argument('--warm-run', type=int, required=True)
    parser.add_argument('--window-size', type=int, required=True)
    parser.add_argument('--fit-run-number', type=int, required=True)

    args = parser.parse_args()

    rjsd_stops = sorted(args.rjsd_stops, reverse=True) if len(args.rjsd_stops) != 0 else [0.0]

    warm_run = args.warm_run
    window_size = args.window_size
    fit_run_number = args.fit_run_number

    ip_dirpath = pathlib.Path(args.ip_dirpath)
    op_dirpath = pathlib.Path(args.op_dirpath)

    print(f'This I Dir Name: {ip_dirpath}')
    print(f'This O Dir Name: {op_dirpath}')
    rjsd_stop_s = list()
    rjsd_stop_i = list()
    for rjsd_stop in rjsd_stops:
        print(f' -> RJSD = {rjsd_stop}')
        all_rjsds = load_pickle(ip_dirpath.joinpath('All_rJSDs.pickle'))
        all_times = load_pickle(ip_dirpath.joinpath('All_Times.pickle'))
        inference_yes, swiped_inference_all_times, total_yes, swiped_total_all_times = swipe_all_times(rjsd_stop, all_rjsds, all_times, fit_run_number, window_size, warm_run)
        rjsd_stop_s.append((rjsd_stop, dict(inference_yes=inference_yes, inference=swiped_inference_all_times, total_yes=total_yes, total=swiped_total_all_times)))
        rjsd_stop_i.append((rjsd_stop, dict(inference_yes=inference_yes, inference=len(swiped_inference_all_times), total_yes=total_yes, total=len(swiped_total_all_times))))
        if inference_yes:
            print(f'    Inference - {len(swiped_inference_all_times)}')
        if total_yes:
            print(f'    Total - {len(swiped_total_all_times)}')
    save_pickle(rjsd_stop_s, op_dirpath.joinpath('rjsd2time_s.pickle'))
    save_pickle(rjsd_stop_i, op_dirpath.joinpath('rjsd2time_i.pickle'))
    print(f'This Done')