#!/usr/bin/env bash
# Download the TimesFM 3.0 checkpoint into models/timesfm-3.0.
# Weights are non-commercial (timesfm-non-commercial-license-v1.0).
set -euo pipefail

dest="${1:-models/timesfm-3.0}"
base="https://huggingface.co/google/timesfm-3.0-pytorch/resolve/main"
mkdir -p "$dest"

for file in config.json model.safetensors; do
  if [ ! -s "$dest/$file" ]; then
    echo "downloading $file"
    curl -L --fail --retry 3 -o "$dest/$file" "$base/$file"
  fi
done
ls -lh "$dest"
