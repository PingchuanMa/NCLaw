from pathlib import Path
import subprocess

from nclaw.constants import SHAPE_ENVS, ENVS, SEEDS, EPOCHS, RENDER, PYTHON_PATH
from nclaw.utils import get_root, get_script_parser, dict_to_hydra


def main():
    root = get_root(__file__)
    python_path = 'python' if PYTHON_PATH is None else PYTHON_PATH

    # ============ start ============

    base_args, unknown = get_script_parser().parse_known_args()
    base_args = vars(base_args)
    base_cmds = [python_path, root / 'train.py']

    mode = 'train'
    quality = 'low'

    elasticity = 'invariant_full_meta'
    plasticity = 'invariant_full_meta'

    for env in ENVS:

        name = Path(env) / mode / f'{elasticity}-{plasticity}'

        args = base_args | {
            'env': env,
            'env/blob/material/elasticity': elasticity,
            'env/blob/material/plasticity': plasticity,
            'env.blob.material.elasticity.requires_grad': True,
            'env.blob.material.plasticity.requires_grad': True,
            'render': RENDER,
            'sim': quality,
            'name': name,
        }

        cmds = base_cmds + dict_to_hydra(args)

        subprocess.run([str(cmd) for cmd in cmds], shell=False)

    # ============= end =============

if __name__ == '__main__':
    main()
