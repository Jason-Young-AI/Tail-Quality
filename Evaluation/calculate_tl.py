import numpy
import pickle
import pathlib
import argparse

from .tail_latency import tail_latency


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate Tail Quality")
    parser.add_argument('--alltime-filepath', type=str, required=True)
    parser.add_argument('--percentiles', type=float, nargs='*')
    args = parser.parse_args()

    percentiles = numpy.array(args.percentiles, dtype=float).tolist()
    alltime_filepath = pathlib.Path(args.alltime_filepath)

    all_inference_time = list()
    all_total_time = list()

    with open(alltime_filepath, 'rb') as alltime_file:
        alltime = pickle.load(alltime_file)
        total_round = 0
        for round_inference_time, round_total_time in zip(alltime['inference'], alltime['total']):
            total_round += 1
            for batch_id, batch_time in round_inference_time.items():
                all_inference_time.append(batch_time)
            for batch_id, batch_time in round_total_time.items():
                all_total_time.append(batch_time)

    assert len(all_inference_time) == len(all_total_time)

    inference_tail_latency = tail_latency(all_inference_time, percentiles)
    total_tail_latency = tail_latency(all_total_time, percentiles)

    print(f'Total Round: {total_round} | All Latency(Second) at Percentile(%)')
    for p, i, t in zip(percentiles, inference_tail_latency, total_tail_latency):
        print(f'{p} %\t: {i}\t {t}')