# Neural Constitutive Laws

[![Website](https://img.shields.io/badge/Project%20Page-Open-yellowgreen.svg)](https://sites.google.com/view/nclaw) [![arXiv](https://img.shields.io/badge/arXiv-2304.14369-b31b1b.svg)](https://arxiv.org/abs/2304.14369)

Learning Neural Constitutive Laws from Motion Observations for Generalizable PDE Dynamics

ICML 2023 / [Website](https://sites.google.com/view/nclaw) / [arXiv](https://arxiv.org/abs/2304.14369)

```
@inproceedings{ma2023learning,
    title={Learning Neural Constitutive Laws from Motion Observations for Generalizable PDE Dynamics},
    author={Ma, Pingchuan and Chen, Peter Yichen and Deng, Bolei and Tenenbaum, Joshua B and Du, Tao and Gan, Chuang and Matusik, Wojciech},
    booktitle={International Conference on Machine Learning},
    year={2023},
    organization={PMLR}
}
```

## Prerequisites

This codebase is tested using the enviroment with following key packages:

- Ubuntu 20.04
- CUDA 11.7
- GCC 11
- Python 3.10.11
- PyTorch 2.0.1
- Warp 0.6.1

## Installation

Prepare conda environment with proper Python:

```bash
conda create -n nclaw python=3.10  # create env
conda activate nclaw               # activate env
```

Install required packages:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install numpy scipy pyvista hydra-core trimesh einops tqdm psutil tensorboard -c defaults -c conda-forge
```

Compile `warp` and install `nclaw`:

```bash
bash ./build.sh
```

## Experiments

Generate dataset:

```bash
python experiments/scripts/dataset/main.py
```

Train NCLaw:

```bash
python experiments/scripts/train/invariant_full_meta-invariant_full_meta.py
```

Evaluate NCLaw:

```bash
# Reconstruction
python experiments/scripts/eval/dataset.py --gt

# Generalization
python experiments/scripts/eval/time.py --gt  # (a) time
python experiments/scripts/eval/vel.py --gt   # (b) velocity
python experiments/scripts/eval/shape.py --gt # (c) geometry
python experiments/scripts/eval/slope.py      # (d) boundary

# Extreme
python experiments/scripts/eval/large.py   # (a) one-million
python experiments/scripts/eval/contact.py # (b) collision

# Multi-physics
python experiments/scripts/eval/pool.py    # (a) coupled-physics
python experiments/scripts/eval/melting.py # (b) phase-transition
```
