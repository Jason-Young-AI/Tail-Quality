#!/bin/bash

get_absolute_path() {
  local relative_path="$1"
  echo "$(cd "$(dirname "$relative_path")"; pwd)/$(basename "$relative_path")"
}

SCRIPTS_ROOT=$(get_absolute_path ".")

cd ../../

python -m Evaluation.calculate_tq --help

cd ${SCRIPTS_ROOT}