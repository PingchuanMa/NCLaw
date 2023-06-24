from typing import Union, Optional
from pathlib import Path
import sys
import shutil
from argparse import ArgumentParser
import numpy as np
import torch
import json


def get_package_root() -> Path:
    return Path(__file__).resolve().parent


def sample_vel(cfg, seed: Optional[int] = None):
    if seed is None:
        seed = cfg.seed
    rng = np.random.Generator(np.random.PCG64(seed))

    lin_dir = rng.uniform(-1, 1, size=3)
    if lin_dir[1] > 0:
        lin_dir[1] = -lin_dir[1]
    lin_dir /= np.linalg.norm(lin_dir)

    lin_mag = rng.uniform(*cfg.lin_vel_bound)
    lin_vel = lin_dir * lin_mag

    ang_vel = rng.uniform(*cfg.ang_vel_bound, size=3)

    return lin_vel, ang_vel


def clean_state(path: Path):
    # comment it out if there is space
    # shutil.rmtree(path, ignore_errors=True)
    pass


def mkdir(path: Path, resume=False, overwrite=False) -> None:

    while True:
        if overwrite:
            if path.is_dir():
                print('overwriting directory ({})'.format(path))
            shutil.rmtree(path, ignore_errors=True)
            path.mkdir(parents=True, exist_ok=True)
            return
        elif resume:
            print('resuming directory ({})'.format(path))
            path.mkdir(parents=True, exist_ok=True)
            return
        else:
            if path.exists():
                feedback = input('target directory ({}) already exists, overwrite? [Y/r/n] '.format(path))
                ret = feedback.casefold()
            else:
                ret = 'y'
            if ret == 'n':
                sys.exit(0)
            elif ret == 'r':
                resume = True
            elif ret == 'y':
                overwrite = True


def get_root(path: Union[str, Path], name: str = '.root') -> Path:
    root = Path(path).resolve()
    while not (root / name).is_file():
        root = root.parent
    return root


def get_script_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('-y', dest='overwrite', action='store_true', help='overwrite')
    parser.add_argument('-r', dest='resume', action='store_true', help='resume')
    parser.add_argument('-g', '--gpu', dest='gpu', type=int, default=0, help='gpu device')
    parser.add_argument('-c', '--cpu', dest='cpu', type=int, default=0, help='cpu offset')
    return parser


def dict_to_hydra(d: dict) -> list[str]:
    return [f'{k}={v}' for k, v in d.items()]


def diff_mse(src: Path, tar: Path, skip_frame=5):
    losses = []
    for i, path in enumerate(sorted((src / 'state').glob('*.pt'))):
        if i % skip_frame != 0:
            continue
        x1 = torch.load(path, map_location='cpu')['x']
        x2 = torch.load(tar / 'state' / path.name, map_location='cpu')['x']
        loss = torch.nn.functional.mse_loss(x1, x2).item()
        losses.append(loss)
    mse = sum(losses) / len(losses)

    info_path = src / 'info.json'
    info = dict()
    if info_path.is_file():
        with info_path.open('r') as f:
            info = json.load(f)

    info['mse'] = mse
    with info_path.open('w') as f:
        json.dump(info, f)

    return info


if __name__ == '__main__':
    root = get_package_root().parent
    state_paths = list(root.glob('**/state'))
    for state_path in state_paths:
        shutil.rmtree(state_path)
