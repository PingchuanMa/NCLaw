from pathlib import Path
import random
from tqdm import tqdm, trange

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn
import warp as wp

import nclaw
from nclaw.material import ComposeMaterial
from nclaw.sim import MPMModelBuilder, MPMStateInitializer, MPMStaticsInitializer, MPMInitData, MPMForwardSim
from nclaw.utils import get_root, sample_vel

root: Path = get_root(__file__)


@torch.no_grad()
@hydra.main(version_base='1.2', config_path=str(root / 'configs'), config_name='eval')
def main(cfg: DictConfig):

    print(OmegaConf.to_yaml(cfg, resolve=True))

    # init

    seed = cfg.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    wp.init()
    wp_device = wp.get_device(f'cuda:{cfg.gpu}')
    wp.ScopedTimer.enabled = False
    wp.set_module_options({'fast_math': False})

    torch_device = torch.device(f'cuda:{cfg.gpu}')
    torch.backends.cudnn.benchmark = True

    requires_grad = False

    # path

    log_root: Path = root / 'log'
    exp_root: Path = log_root / cfg.name
    state_root: Path = exp_root / 'state'
    nclaw.utils.mkdir(state_root, overwrite=cfg.overwrite, resume=cfg.resume)
    OmegaConf.save(cfg, exp_root / 'hydra.yaml', resolve=True)

    # warp

    model = MPMModelBuilder().parse_cfg(cfg.sim).finalize(wp_device, requires_grad)
    sim = MPMForwardSim(model)
    state_initializer = MPMStateInitializer(model)
    statics_initializer = MPMStaticsInitializer(model)

    # env

    elasticities = []
    plasticities = []

    for blob, blob_cfg in sorted(cfg.env.items()):

        if blob in ['name', 'start', 'end']:
            continue

        # material

        for material in ['material0', 'material1']:

            elasticity: nn.Module = getattr(nclaw.material, blob_cfg[material].elasticity.cls)(blob_cfg[material].elasticity)
            plasticity: nn.Module = getattr(nclaw.material, blob_cfg[material].plasticity.cls)(blob_cfg[material].plasticity)

            if ckpt_path := blob_cfg[material].ckpt:

                ckpt = torch.load(log_root / ckpt_path, map_location=torch_device)
                elasticity.load_state_dict(ckpt['elasticity'])
                plasticity.load_state_dict(ckpt['plasticity'])

            elasticities.append(elasticity)
            plasticities.append(plasticity)

        init_data = MPMInitData.get(blob_cfg)

        if blob_cfg.vel.random:
            lin_vel, ang_vel = sample_vel(blob_cfg.vel)
        else:
            lin_vel = np.array(blob_cfg.vel.lin_vel)
            ang_vel = np.array(blob_cfg.vel.ang_vel)
        init_data.set_lin_vel(lin_vel)
        init_data.set_ang_vel(ang_vel)

        state_initializer.add_group(init_data)
        statics_initializer.add_group(init_data)

    state, sections = state_initializer.finalize()
    statics = statics_initializer.finalize()
    x, v, C, F, stress = state.to_torch()

    types = torch.zeros(sum(sections), dtype=torch.int)
    for i, (left, right) in enumerate(zip(np.cumsum(sections)[:-1], np.cumsum(sections)[1:])):
        types[left:right] = i
    torch.save(dict(x=x, v=v, C=C, F=F, stress=stress, sections=sections, types=types), state_root / f'{0:04d}.pt')

    material_sections = []

    for step in range(1, cfg.env.start + 1):
        material_sections.append([sections[0], 0, sections[1], 0])

    ys = x.cpu().clone().detach().numpy()[:, 1]
    high = ys.max()
    low = ys.min()
    for step in range(cfg.env.start + 1, cfg.env.end + 1):
        progress = (step - cfg.env.start) / (cfg.env.end - cfg.env.start)
        y_thredhold = low + progress * (high - low)
        idx1, idx2 = 0, 0
        while ys[idx1] > y_thredhold and idx1 < sections[0]:
            idx1 += 1
        while ys[idx2 + sections[0]] > y_thredhold and idx2 < sections[1]:
            idx2 += 1

        material_sections.append([idx1, sections[0] - idx1, idx2, sections[1] - idx2])

    for step in range(cfg.env.end + 1, cfg.sim.num_steps + 1):
        material_sections.append([0, sections[0], 0, sections[1]])

    elasticity = ComposeMaterial(elasticities, None)
    elasticity.to(torch_device)
    elasticity.requires_grad_(requires_grad)
    elasticity.eval()

    plasticity = ComposeMaterial(plasticities, None)
    plasticity.to(torch_device)
    plasticity.requires_grad_(requires_grad)
    plasticity.eval()

    for step in trange(cfg.sim.num_steps):
        elasticity.update_sections(material_sections[step])
        plasticity.update_sections(material_sections[step])

        types = torch.zeros(sum(material_sections[step]), dtype=torch.int)
        for i, (left, right) in enumerate(zip(np.cumsum(sections)[:-1], np.cumsum(sections)[1:])):
            types[left:right] = i % 2

        stress = elasticity(F)
        state.from_torch(stress=stress)
        x, v, C, F = sim(statics, state)
        F = plasticity(F)
        state.from_torch(F=F)
        statics_initializer.update(statics, step)
        torch.save(dict(x=x, v=v, C=C, F=F, stress=stress, sections=sections, types=types), state_root / f'{step + 1:04d}.pt')


if __name__ == '__main__':
    main()
