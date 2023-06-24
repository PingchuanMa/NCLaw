from pathlib import Path
import subprocess
import os
from argparse import ArgumentParser

from nclaw.constants import SHAPE_ENVS, ENVS, SEEDS, EPOCHS, RENDER, PYTHON_PATH
from nclaw.utils import get_root, get_script_parser, dict_to_hydra, clean_state
from nclaw.ffmpeg import cat_videos


def main():
    root = get_root(__file__)
    python_path = 'python' if PYTHON_PATH is None else PYTHON_PATH

    # ============ start ============

    base_args, unknown = get_script_parser().parse_known_args()
    base_args = vars(base_args)

    my_env = os.environ.copy()
    my_env['CUDA_VISIBLE_DEVICES'] = str(base_args['gpu'])

    parser = ArgumentParser()
    parser.add_argument('--skip', dest='skip', action='store_true')
    parser.add_argument('--render', dest='render', nargs='+', choices=['pv'], default=['pv'])
    parser.add_argument('--video', dest='video', nargs='+', choices=['pv'], default=None)
    extra_args, _ = parser.parse_known_args(unknown)
    if extra_args.video is None:
        extra_args.video = extra_args.render

    quality = 'high'
    env = 'contact'

    elasticity = 'invariant_full_meta'
    plasticity = 'invariant_full_meta'

    epoch_args_list = []
    gt_args_list = []

    name = Path(env)

    subargs = base_args | {
        'env': env,
        'render': RENDER,
        'sim': quality,
    }

    args = subargs | {
        'name': Path(env) / 'gt'
    }
    gt_args_list.append(args)

    for epoch in EPOCHS:

        exp_name = name / 'epoch' / f'{epoch:04d}'

        args = subargs | {
            'env/blob/material/elasticity@env.bunny.material.elasticity': elasticity,
            'env/blob/material/plasticity@env.bunny.material.plasticity': plasticity,
            'env.bunny.material.ckpt': Path('jelly') / 'train' / f'{elasticity}-{plasticity}' / 'ckpt' / f'{epoch:04d}.pt',
            'env/blob/material/elasticity@env.spot.material.elasticity': elasticity,
            'env/blob/material/plasticity@env.spot.material.plasticity': plasticity,
            'env.spot.material.ckpt': Path('jelly') / 'train' / f'{elasticity}-{plasticity}' / 'ckpt' / f'{epoch:04d}.pt',
            'env/blob/material/elasticity@env.duck.material.elasticity': elasticity,
            'env/blob/material/plasticity@env.duck.material.plasticity': plasticity,
            'env.duck.material.ckpt': Path('jelly') / 'train' / f'{elasticity}-{plasticity}' / 'ckpt' / f'{epoch:04d}.pt',
            'env/blob/material/elasticity@env.blub.material.elasticity': elasticity,
            'env/blob/material/plasticity@env.blub.material.plasticity': plasticity,
            'env.blub.material.ckpt': Path('jelly') / 'train' / f'{elasticity}-{plasticity}' / 'ckpt' / f'{epoch:04d}.pt',
            'name': exp_name
        }
        epoch_args_list.append(args)

    for args in gt_args_list + epoch_args_list:

        # ============ eval ============

        if not extra_args.skip:
            base_cmds = [python_path, root / 'eval.py']
            cmds = base_cmds + dict_to_hydra(args)
            subprocess.run([str(cmd) for cmd in cmds], shell=False)

        # ============ render ============

        for render in extra_args.render:

            base_cmds = [python_path, root / 'render' / 'cube' / f'{render}.py']
            cmds = base_cmds + dict_to_hydra(args)
            subprocess.run([str(cmd) for cmd in cmds], shell=False, env=my_env)

        clean_state(root / 'log' / args['name'] / 'state')

    for render in extra_args.video:
        videos = [root / 'log' / args['name'] / f'{render}.mp4' for args in (epoch_args_list + gt_args_list)]
        cat_videos(videos, root / 'log' / name / 'videos' / f'{render}.mp4')

    # ============= end =============

if __name__ == '__main__':
    main()
