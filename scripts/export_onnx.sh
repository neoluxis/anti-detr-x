#!/usr/bin/env bash

set -euo pipefail

root_dir="${1:-runs/detect/cst-sample-5k1k1k-s100}"

shopt -s nullglob
models=("${root_dir}"/*/weights/best.pt)
shopt -u nullglob

if [[ ${#models[@]} -eq 0 ]]; then
  echo "No best.pt files found under ${root_dir}/*/weights/" >&2
  exit 1
fi

for model in "${models[@]}"; do
  echo "Exporting ${model}"
  uv run yolo export model="${model}" format=onnx
done
