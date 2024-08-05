import json
import pickle
import pathlib

from typing import Any, Literal


def load_json(filepath: pathlib.Path) -> Any:
    info = None
    with open(filepath, 'r') as file:
        info = json.load(file)

    return info


def load_pickle(filepath: pathlib.Path) -> Any:
    info = None
    with open(filepath, 'rb') as file:
        info = pickle.load(file)

    return info


def save_pickle(object: Any, filepath: pathlib.Path):
    with open(filepath, 'wb') as file:
        pickle.dump(object, file)