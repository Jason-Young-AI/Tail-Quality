from typing import Any


def expand_indexed_batches(indexed_batches: dict[str, Any]) -> tuple[list[Any], list[int]]:
    indexed_batches = sorted([(int(index), batch) for index, batch in indexed_batches.items()], key=lambda x: x[0])
    sizes = list()
    expanded = list()
    for index, batch in indexed_batches:
        if isinstance(batch, list):
            sizes.append(len(batch))
            expanded.extend(batch)
        else:
            sizes.append(1)
            expanded.append(batch)
    
    return expanded, sizes


def expand_round_time(round_time: dict[str, float], batch_sizes: list[int]) -> list[float]:
    round_time = sorted([(int(index), batch_time) for index, batch_time in round_time.items()], key=lambda x: x[0])
    assert len(round_time) == len(batch_sizes)
    expanded = list()
    for (index, batch_time), batch_size in zip(round_time, batch_sizes):
        expanded.extend([batch_time for _ in range(batch_size)])
    return expanded