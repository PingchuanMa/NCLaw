from pathlib import Path
import subprocess
import os
from argparse import ArgumentParser

from nclaw.constants import SHAPE_ENVS, ENVS, SEEDS, EPOCHS, RENDER, PYTHON_PATH
from nclaw.utils import get_root, get_script_parser, dict_to_hydra, clean_state


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
    parser.add_argument('--render', dest='render', nargs='+', choices=['pv'], default=[])
    extra_args, _ = parser.parse_known_args(unknown)

    mode = 'dataset'

    quality = 'low'

    for env in ENVS:

        name = Path(env) / mode

        args = base_args | {
            'env': env,
            'render': RENDER,
            'sim': quality,
            'name': name,
            'dataset': True,
        }

        if not extra_args.skip:
            base_cmds = [python_path, root / 'eval.py']
            cmds = base_cmds + dict_to_hydra(args)
            subprocess.run([str(cmd) for cmd in cmds], shell=False)

        for render in extra_args.render:

            base_cmds = [python_path, root / 'render' / 'cube' / f'{render}.py']
            cmds = base_cmds + dict_to_hydra(args)
            subprocess.run([str(cmd) for cmd in cmds], shell=False, env=my_env)


if __name__ == '__main__':
    main()
