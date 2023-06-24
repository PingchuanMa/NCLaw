#! /bin/bash

(
    pip uninstall -y warp warp-lang &&
    cd ./third_party/warp &&
    python build_lib.py --mode release --fast_math True &&
    python build_exports.py &&
    pip install . -v
)

pip install -e . -v
