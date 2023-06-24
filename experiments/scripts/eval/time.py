from pathlib import Path
import subprocess
import os
from argparse import ArgumentParser

from nclaw.constants import SHAPE_ENVS, ENVS, SEEDS, EPOCHS, RENDER, PYTHON_PATH
from nclaw.utils import get_root, get_script_parser, dict_to_hydra, clean_state, diff_mse
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
    parser.add_argument('--gt', dest='gt', action='store_true')
    parser.add_argument('--render', dest='render', nargs='+', choices=['pv'], default=['pv'])
    parser.add_argument('--video', dest='video', nargs='+', choices=['pv'], default=None)
    extra_args, _ = parser.parse_known_args(unknown)
    if extra_args.video is None:
        extra_args.video = extra_args.render

    gt_mode = 'time'
    mode = 'train'

    quality = 'low'

    for env in ENVS:

        gt_args_list = []
        subargs = base_args | {
            'env': env,
            'render': RENDER,
            'sim': quality,
            'sim.num_steps': 5000 if env == 'water' else 2000
        }

        args = subargs | {
            'name': Path(env) / gt_mode
        }
        gt_args_list.append(args)

        if extra_args.gt:
            for args in gt_args_list:

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

        for elasticity, plasticity in [
                ['invariant_full_meta', 'invariant_full_meta'],
                # ['sysid', 'sysid'],
                # ['sysid-blind', 'sysid-blind'],
                # ['SVD_meta', 'identity'],
                # ['supervised', 'supervised'],
                # ['spline_meta', 'identity'],
                ]:

            special = None
            if elasticity == 'sysid' or plasticity == 'sysid':
                special = 'sysid'
            elif elasticity == 'sysid-blind' or plasticity == 'sysid-blind':
                special = 'sysid-blind'
            elif elasticity == 'supervised' or plasticity == 'supervised':
                special = 'supervised'

            epoch_args_list = []

            if special is None:
                name = Path(env) / mode / f'{elasticity}-{plasticity}'
            else:
                name = Path(env) / mode / special

            for epoch in EPOCHS:

                exp_name = name / gt_mode / f'{epoch:04d}'
                ckpt_name = name / 'ckpt' / f'{epoch:04d}.pt'

                if special is None:
                    args = subargs | {
                        'env/blob/material/elasticity': elasticity,
                        'env/blob/material/plasticity': plasticity,
                        'env.blob.material.ckpt': ckpt_name,
                        'name': exp_name,
                    }
                elif special == 'sysid':
                    args = subargs | {
                        'env.blob.material.ckpt': ckpt_name,
                        'name': exp_name,
                    }
                elif special == 'sysid-blind':
                    args = subargs | {
                        'env/blob/material/elasticity': 'sigma_plasticine',
                        'env/blob/material/plasticity': 'von_mises_plasticine',
                        'env.blob.material.ckpt': ckpt_name,
                        'name': exp_name,
                    }
                elif special == 'supervised':
                    args = subargs | {
                        'env/blob/material/elasticity': 'invariant_full_meta',
                        'env/blob/material/plasticity': 'invariant_full_meta',
                        'env.blob.material.ckpt': ckpt_name,
                        'name': exp_name,
                    }

                epoch_args_list.append(args)

            for args in epoch_args_list:

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
                cat_videos(videos, root / 'log' / name / 'videos' / f'{gt_mode}-{render}.mp4')

            for args in epoch_args_list:
                info = diff_mse(root / 'log' / args['name'], root / 'log' / gt_args_list[0]['name'], skip_frame=5)
                print('{}: {}'.format(args['name'], info['mse']))

    # ============= end =============

if __name__ == '__main__':
    main()
