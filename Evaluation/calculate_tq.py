import json
import numpy
import pickle
import pathlib
import argparse

from typing import Any, Literal

from metrics.detr import get_detr_metrics
from metrics.vicuna import get_vicuna_metrics
from metrics.light_gcn import get_light_gcn_metrics
from metrics.hybrid_nets import get_hybrid_nets_metrics
from metrics.mobile_net import get_mobile_net_metrics
from metrics.emotion_flow import get_emotion_flow_metrics

from tail_quality import tail_quality


quality_calculation_functions = dict(
    detr = get_detr_metrics,
    vicuna = get_vicuna_metrics,
    light_gcn = get_light_gcn_metrics,
    hybrid_nets = get_hybrid_nets_metrics,
    mobile_net = get_mobile_net_metrics,
    emotion_flow = get_emotion_flow_metrics,
)


def load_info(filepath: pathlib.Path, filetype: Literal['json', 'pickle']) -> Any:
    info = None
    if filetype == 'json':
        with open(filepath, 'r') as file:
            info = json.load(file)
    if filetype == 'pickle':
        with open(filepath, 'rb') as file:
            info = pickle.load(file)

    return info


def expand_indexed_batches(indexed_batches: dict[str, list]) -> tuple[list[Any], list[int]]:
    indexed_batches = sorted([(int(index), batch) for index, batch in indexed_batches.items()], key=lambda x: x[0])
    sizes = list()
    expanded = list()
    for index, batch in indexed_batches:
        sizes.append(len(batch))
        expanded.extend(batch)
    
    return expanded, sizes


def expand_round_time(round_time: dict[str, float], batch_sizes: list[int]) -> list[float]:
    round_time = sorted([(int(index), batch_time) for index, batch_time in round_time.items()], key=lambda x: x[0])
    assert len(round_time) == len(batch_sizes)
    expanded = list()
    for (index, batch_time), batch_size in zip(round_time, batch_sizes):
        expanded.extend([batch_time for _ in range(batch_size)])
    return expanded


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate Tail Quality")
    parser.add_argument('-t', '--alltime-filepath', type=str, required=True)
    parser.add_argument('-p', '--results-filepath', type=str, required=True)
    parser.add_argument('-g', '--goldens-filepath', type=str, required=True)

    parser.add_argument('-s', '--specific-thresholds', type=float, nargs='*')
    parser.add_argument('--specific-filepath', type=str, default=None)

    parser.add_argument('-l', '--multihop-thresholds-min', type=float, default=None)
    parser.add_argument('-r', '--multihop-thresholds-max', type=float, default=None)
    parser.add_argument('-h', '--multihop-thresholds-hop', type=float, default=None)
    parser.add_argument('--multihop-filepath', type=str, default=None)

    parser.add_argument('--alltime-type', type=str, choices=['inference', 'total'], required=True)

    parser.add_argument('--results-filetype', type=str, choices=['json', 'pickle'], required=True)
    parser.add_argument('--goldens-filetype', type=str, choices=['json', 'pickle'], required=True)

    parser.add_argument('--task-name', type=str, choices=quality_calculation_functions.keys(), required=True)

    parser.add_argument('--worker-number', type=int, default=8)
    args = parser.parse_args()

    specific_thresholds = numpy.array(args.specific_thresholds, dtype=float).tolist()
    if len(specific_thresholds) != 0:
        assert args.specific_filepath is not None
        specific_filepath = pathlib.Path(args.specific_filepath)

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

    if len(specific_thresholds) == 0 and len(multihop_thresholds) == 0:
        print(f'Not Specify Any Thresholds!')
    else:
        results_filepath = pathlib.Path(args.results_filepath)
        results = load_info(results_filepath, args.results_filetype)
        results, results_batch_sizes = expand_indexed_batches(results)
        
        goldens_filepath = pathlib.Path(args.goldens_filepath)
        goldens = load_info(goldens_filepath, args.goldens_filetype)
        goldens, goldens_batch_sizes = expand_indexed_batches(goldens)

        assert len(results) == len(goldens)
        for result_batch_size, golden_batch_size in zip(results_batch_sizes, goldens_batch_sizes):
            assert result_batch_size == golden_batch_size

        alltime_filepath = pathlib.Path(args.alltime_filepath)
        alltime = load_info(alltime_filepath, 'pickle')[args.alltime_type]

        multiple_inference_times = list()
        for round_time in alltime:
            total_round = 0
            round_time = expand_round_time(round_time, results_batch_sizes)
            multiple_inference_times.append(round_time)

        print(f'Calculating Specific Tail Qualities ... ')
        specific_tq = tail_quality(quality_calculation_functions[args.task_name], goldens, results, multiple_inference_times, specific_thresholds, args.worker_number)
        specific_thres2tq = [(thres, tq) for thres, tq in zip(specific_thresholds, specific_tq)]
        with open(specific_filepath, 'wb') as specific_file:
            pickle.dump(specific_thres2tq, specific_file)
        print(f'Done')
        print(f'Calculating Multihop Tail Qualities ... ')
        multihop_tq = tail_quality(quality_calculation_functions[args.task_name], goldens, results, multiple_inference_times, multihop_thresholds, args.worker_number)
        multihop_thres2tq = [(thres, tq) for thres, tq in zip(multihop_thresholds, multihop_tq)]
        with open(multihop_filepath, 'wb') as multihop_file:
            pickle.dump(multihop_thres2tq, multihop_file)
        print(f'Done')