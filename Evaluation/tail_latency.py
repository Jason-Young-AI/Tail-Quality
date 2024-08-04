import numpy


def tail_latency(inference_times: list[float], percentiles: list[float]) -> list[float]:
    # inference_times: [time_1 s, time_2 s, ..., time_n s] [0.004s, 0.5s, ..., 0.06s] (unit is second)
    # percentiles: [percentile_1 %, percentile_2 %, ..., percentile_m %] [0%, 95%, ..., 100%] (unit is %)
    return numpy.percentile(inference_times, percentiles, method='inverted_cdf').tolist()