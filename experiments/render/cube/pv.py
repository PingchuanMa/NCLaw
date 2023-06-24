from pathlib import Path
from tqdm import tqdm
import time
import psutil
import random
import fcntl
import os
import subprocess
import tempfile
from random import randint
from errno import EACCES

import hydra
from omegaconf import DictConfig
import numpy as np
import torch
import pyvista as pv

import nclaw
from nclaw.utils import get_root
from nclaw.ffmpeg import make_video
from nclaw.render import get_camera

root: Path = get_root(__file__)


class Xvfb(object):

    # Maximum value to use for a display. 32-bit maxint is the
    # highest Xvfb currently supports
    MAX_DISPLAY = 2147483647
    SLEEP_TIME_BEFORE_START = 0.1

    def __init__(
            self, width=800, height=680, colordepth=24,
            tempdir=None, display=None, **kwargs):
        self.width = width
        self.height = height
        self.colordepth = colordepth
        self._tempdir = tempdir or tempfile.gettempdir()
        self.new_display = display

        if not self.xvfb_exists():
            msg = (
                'Can not find Xvfb. Please install it with:\n'
                '   sudo apt install libgl1-mesa-glx xvfb')
            raise EnvironmentError(msg)

        self.extra_xvfb_args = ['-screen', '0', '{}x{}x{}'.format(
                                self.width, self.height, self.colordepth)]

        for key, value in kwargs.items():
            self.extra_xvfb_args += ['-{}'.format(key), value]

        if 'DISPLAY' in os.environ:
            self.orig_display = os.environ['DISPLAY'].split(':')[1]
        else:
            self.orig_display = None

        self.proc = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        if self.new_display is not None:
            if not self._get_lock_for_display(self.new_display):
                raise ValueError(f'Could not lock display :{self.new_display}')
        else:
            self.new_display = self._get_next_unused_display()
        display_var = ':{}'.format(self.new_display)
        self.xvfb_cmd = ['Xvfb', display_var] + self.extra_xvfb_args
        with open(os.devnull, 'w') as fnull:
            self.proc = subprocess.Popen(
                self.xvfb_cmd, stdout=fnull, stderr=fnull, close_fds=True)
        # give Xvfb time to start
        time.sleep(self.__class__.SLEEP_TIME_BEFORE_START)
        ret_code = self.proc.poll()
        if ret_code is None:
            self._set_display_var(self.new_display)
        else:
            self._cleanup_lock_file()
            raise RuntimeError(
                f'Xvfb did not start ({ret_code}): {self.xvfb_cmd}')

    def stop(self):
        try:
            if self.orig_display is None:
                del os.environ['DISPLAY']
            else:
                self._set_display_var(self.orig_display)
            if self.proc is not None:
                try:
                    self.proc.terminate()
                    self.proc.wait()
                except OSError:
                    pass
                self.proc = None
        finally:
            self._cleanup_lock_file()

    def xvfb_exists(self):
        """Check that Xvfb is available on PATH and is executable."""
        paths = os.environ['PATH'].split(os.pathsep)
        return any(os.access(os.path.join(path, 'Xvfb'), os.X_OK)
                   for path in paths)

    def _cleanup_lock_file(self):
        '''
        This should always get called if the process exits safely
        with Xvfb.stop() (whether called explicitly, or by __exit__).
        If you are ending up with /tmp/X123-lock files when Xvfb is not
        running, then Xvfb is not exiting cleanly. Always either call
        Xvfb.stop() in a finally block, or use Xvfb as a context manager
        to ensure lock files are purged.
        '''
        self._lock_display_file.close()
        try:
            os.remove(self._lock_display_file.name)
        except OSError:
            pass

    def _get_lock_for_display(self, display):
        '''
        In order to ensure multi-process safety, this method attempts
        to acquire an exclusive lock on a temporary file whose name
        contains the display number for Xvfb.
        '''
        tempfile_path = os.path.join(self._tempdir, '.X{0}-lock'.format(display))
        try:
            self._lock_display_file = open(tempfile_path, 'w')
        except PermissionError as e:
            return False
        else:
            try:
                fcntl.flock(self._lock_display_file,
                            fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            else:
                return True

    def _get_next_unused_display(self):
        '''
        Randomly chooses a display number and tries to acquire a lock for this number.
        If the lock could be acquired, returns this number, otherwise choses a new one.
        :return: free display number
        '''
        while True:
            rand = randint(1, self.__class__.MAX_DISPLAY)
            if self._get_lock_for_display(rand):
                return rand
            else:
                continue

    def _set_display_var(self, display):
        os.environ['DISPLAY'] = ':{}'.format(display)


@torch.no_grad()
def render(cfg):

    seed = cfg.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    render_type = 'pv'

    log_root: Path = root / 'log'
    exp_root: Path = log_root / cfg.name
    state_root: Path = exp_root / 'state'
    image_root: Path = exp_root / render_type
    nclaw.utils.mkdir(image_root, overwrite=cfg.overwrite, resume=cfg.resume)

    scale = cfg.sim.num_grids / (cfg.sim.num_grids - 2 * cfg.render.bound)

    plotter = pv.Plotter(lighting='three lights', off_screen=True, window_size=(cfg.render.width, cfg.render.height))
    plotter.set_background('white')
    plotter.camera_position = get_camera()
    plotter.enable_shadows()

    ckpt_paths = list(sorted(state_root.glob('*.pt'), key=lambda x: int(x.stem)))
    for i, path in enumerate(tqdm(ckpt_paths, desc=render_type)):

        if i % cfg.render.skip_frame != 0:
            continue

        ckpt = torch.load(path, map_location='cpu')
        x_sections = np.split((ckpt['x'].cpu().detach().numpy() - 0.5) * scale + 0.5, np.cumsum(ckpt['sections']), axis=0)

        j = 0
        for blob, blob_cfg in sorted(cfg.env.items()):
            if blob in ['name', 'start', 'end']:
                continue
            radius = 0.5 * np.power(np.prod(blob_cfg.shape.size) / x_sections[j].shape[0], 1 / 3) * scale
            x_sections[j] = np.clip(x_sections[j], radius, 1 - radius)

            polydata = pv.PolyData(x_sections[j])
            if blob_cfg.span[0] <= i <= blob_cfg.span[1]:
                plotter.add_mesh(polydata, style='points', name=blob, render_points_as_spheres=True, point_size=radius * 1000, color=list(blob_cfg.bsdf_pcd.reflectance.value))
            else:
                plotter.remove_actor(blob)

            j += 1

        plotter.show(auto_close=False, screenshot=str(image_root / f'{i // cfg.render.skip_frame:04d}.png'))

    plotter.close()
    make_video(image_root, exp_root / f'{render_type}.mp4', '%04d.png', cfg.render.fps)


@torch.no_grad()
@hydra.main(version_base='1.2', config_path=str(root / 'configs'), config_name='pv')
def main(cfg: DictConfig):

    with Xvfb():
        render(cfg)


if __name__ == '__main__':
    main()
