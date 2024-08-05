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

from .utils.io import save_pickle, load_pickle
from .tail_quality import tail_quality
from .calculate_tl import get_tail_latency


tasks: dict[str, Task] = dict(
    detr         = DETR,
    vicuna       = Vicuna,
    light_gcn    = LightGCN,
    hybrid_nets  = HybridNets,
    mobile_net   = MobileNet,
    emotion_flow = EmotionFlow,
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate Tail Quality")
    parser.add_argument('-t', '--alltime-filepath', type=str, required=True)
    parser.add_argument('-p', '--results-filepath', type=str, required=True)
    parser.add_argument('-g', '--goldens-filepath', type=str, required=True)

    parser.add_argument('-s', '--specific-thresholds', type=float, nargs='*')
    parser.add_argument('--specific-filepath', type=str, default=None)

    parser.add_argument('-l', '--multihop-thresholds-min', type=float, default=None)
    parser.add_argument('-r', '--multihop-thresholds-max', type=float, default=None)
    parser.add_argument('-o', '--multihop-thresholds-hop', type=float, default=None)
    parser.add_argument('--multihop-filepath', type=str, default=None)

    parser.add_argument('--alltime-type', type=str, choices=['inference', 'total'], required=True)

    parser.add_argument('--task-name', type=str, choices=tasks.keys(), required=True)

    parser.add_argument('--worker-number', type=int, default=8)
    args = parser.parse_args()

    specific_thresholds = numpy.array(args.specific_thresholds, dtype=float).tolist() if args.specific_thresholds is not None else numpy.array([], dtype=float)
    if len(specific_thresholds) != 0:
        assert args.specific_filepath is not None
        specific_filepath = pathlib.Path(args.specific_filepath)
    else:
        specific_filepath = None

    multihop_thresholds_min = args.multihop_thresholds_min
    multihop_thresholds_max = args.multihop_thresholds_max
    multihop_thresholds_hop = args.multihop_thresholds_hop

    if multihop_thresholds_min is not None and multihop_thresholds_max is not None:
        assert multihop_thresholds_min < multihop_thresholds_max
        assert multihop_thresholds_hop is not None
        assert 0 < multihop_thresholds_hop
        multihop_thresholds = numpy.linspace(multihop_thresholds_min, multihop_thresholds_max, multihop_thresholds_hop, dtype=float).tolist()
    else:
        multihop_thresholds = numpy.array([], dtype=float).tolist()
    if len(multihop_thresholds) != 0:
        assert args.multihop_filepath is not None
        multihop_filepath = pathlib.Path(args.multihop_filepath)
    else:
        multihop_filepath = None

    if len(specific_thresholds) == 0 and len(multihop_thresholds) == 0:
        print(f'Not Specify Any Thresholds!')
        if args.specific_filepath is None and args.multihop_filepath is None:
            print(f'Exit!')
            sys.exit(0)
        else:
            alltime = load_pickle(args.alltime_filepath)
            total_round, inference_tail_latency, total_tail_latency = get_tail_latency(alltime, [0, 100, 90, 95, 99, 99.9])
            if args.alltime_type == 'inference':
                tls = inference_tail_latency
            if args.alltime_type == 'total':
                tls = total_tail_latency

            if args.specific_filepath is not None:
                print(f'You Specified Specific Filepath')
                print(f'Now Using Default Specific Thresholds:')
                specific_filepath = pathlib.Path(args.specific_filepath)
                specific_thresholds = tls[2:]
                print(f'Multihop: 90% \t\t | 95% \t\t | 99% \t\t | 99.9%')
                print(f'Multihop: {tls[2]:.12f} \t\t | {tls[3]:.12f} \t\t | {tls[4]:.12f} \t\t | {tls[5]:.12f}')

            if args.multihop_filepath is not None:
                print(f'You Specified Multihop Filepath')
                print(f'Now Using Default Multihop Thresholds:')
                multihop_filepath = pathlib.Path(args.multihop_filepath)
                multihop_thresholds = numpy.linspace(tls[0], tls[1], 1000, dtype=float).tolist()
                print(f'Multihop: Min \t\t | Max \t\t | Hop')
                print(f'Multihop: {tls[0]:.12f} \t\t | {tls[1]:.12f} \t\t | {1000}')

    if specific_filepath is None and multihop_filepath is None:
        print(f'Exit!')
        sys.exit(0)
    else:
        task = tasks[args.task_name]
        results_filepath = pathlib.Path(args.results_filepath)
        goldens_filepath = pathlib.Path(args.goldens_filepath)
        alltime_filepath = pathlib.Path(args.alltime_filepath)
        print(f'Preprocessing All Goldens, Results, and Inference Times ... ')
        goldens, results, multiple_inference_times = task.pre_process(goldens_filepath, results_filepath, alltime_filepath, args.alltime_type)
        print(f'Done')

        if specific_filepath is not None:
            print(f'Calculating Specific Tail Qualities ... ')
            specific_tq = tail_quality(task.get_metrics, goldens, results, multiple_inference_times, specific_thresholds, args.worker_number)
            specific_thres2tq = [(thres, tq) for thres, tq in zip(specific_thresholds, specific_tq)]
            save_pickle(specific_thres2tq, specific_filepath)
            print(f'Done')

        if multihop_filepath is not None:
            print(f'Calculating Multihop Tail Qualities ... ')
            multihop_tq = tail_quality(task.get_metrics, goldens, results, multiple_inference_times, multihop_thresholds, args.worker_number)
            multihop_thres2tq = [(thres, tq) for thres, tq in zip(multihop_thresholds, multihop_tq)]
            save_pickle(multihop_thres2tq, multihop_filepath)
            print(f'Done')
