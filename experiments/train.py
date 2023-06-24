from pathlib import Path
import random
import time
from collections import defaultdict
from tqdm.autonotebook import tqdm, trange

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
import torch.backends.cudnn
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
import warp as wp

import nclaw
from nclaw.train import Teacher
from nclaw.data import MPMDataset
from nclaw.sim import MPMModelBuilder, MPMCacheDiffSim, MPMStaticsInitializer, MPMInitData
from nclaw.utils import get_root

root: Path = get_root(__file__)


@hydra.main(version_base='1.2', config_path=str(root / 'configs'), config_name='train')
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

    # path

    log_root: Path = root / 'log'
    exp_root: Path = log_root / cfg.name
    nclaw.utils.mkdir(exp_root, overwrite=cfg.overwrite, resume=cfg.resume)
    OmegaConf.save(cfg, exp_root / 'hydra.yaml', resolve=True)
    writer = SummaryWriter(exp_root, purge_step=0)

    ckpt_root: Path = exp_root / 'ckpt'
    ckpt_root.mkdir(parents=True, exist_ok=True)

    dataset_root: Path = log_root / cfg.env.blob.material.name / 'dataset'

    # data

    dataset = MPMDataset(dataset_root, torch_device)

    # warp

    model = MPMModelBuilder().parse_cfg(cfg.sim).finalize(wp_device, requires_grad=True)
    sim = MPMCacheDiffSim(model, cfg.sim.num_steps)
    statics_initializer = MPMStaticsInitializer(model)
    init_data = MPMInitData.get(cfg.env.blob)
    statics_initializer.add_group(init_data)
    statics = statics_initializer.finalize()

    # material

    elasticity_requires_grad = cfg.env.blob.material.elasticity.requires_grad
    plasticity_requires_grad = cfg.env.blob.material.plasticity.requires_grad

    elasticity: nn.Module = getattr(nclaw.material, cfg.env.blob.material.elasticity.cls)(cfg.env.blob.material.elasticity)
    elasticity.to(torch_device)
    if len(list(elasticity.parameters())) == 0:
        elasticity_requires_grad = False
    elasticity.requires_grad_(elasticity_requires_grad)
    elasticity.train(True)

    plasticity: nn.Module = getattr(nclaw.material, cfg.env.blob.material.plasticity.cls)(cfg.env.blob.material.plasticity)
    plasticity.to(torch_device)
    if len(list(plasticity.parameters())) == 0:
        plasticity_requires_grad = False
    plasticity.requires_grad_(plasticity_requires_grad)
    plasticity.train(True)

    torch.save({
        'elasticity': elasticity.state_dict(),
        'plasticity': plasticity.state_dict(),
    }, ckpt_root / f'{0:04d}.pt')

    if elasticity_requires_grad:
        elasticity_optimizer = torch.optim.Adam(elasticity.parameters(), lr=cfg.train.elasticity_lr, weight_decay=cfg.train.elasticity_wd)
        elasticity_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=elasticity_optimizer, T_max=cfg.train.num_epochs)
    if plasticity_requires_grad:
        plasticity_optimizer = torch.optim.Adam(plasticity.parameters(), lr=cfg.train.plasticity_lr, weight_decay=cfg.train.plasticity_wd)
        plasticity_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=plasticity_optimizer, T_max=cfg.train.num_epochs)
    teacher = Teacher(cfg)

    criterion = nn.MSELoss()
    criterion.to(torch_device)

    for epoch in trange(cfg.train.num_epochs, position=1):

        if elasticity_requires_grad:
            elasticity_optimizer.zero_grad()
        if plasticity_requires_grad:
            plasticity_optimizer.zero_grad()

        losses = defaultdict(int)

        num_teachers = teacher.num_teachers(cfg.sim.num_steps)
        loss_factor = 1e4 * teacher.loss_factor(cfg.sim.num_steps)

        xt, vt, Ct, Ft, _ = dataset[0]
        for step, is_teacher in enumerate(tqdm(teacher(cfg.sim.num_steps), position=0, leave=False)):
            if is_teacher:
                x, v, C, F = xt, vt, Ct, Ft
            stress = elasticity(F)
            x, v, C, F = sim(statics, step, x, v, C, F, stress)
            F = plasticity(F)
            xt, vt, Ct, Ft, _ = dataset[step + 1]
            loss_acc = criterion(x, xt) * loss_factor
            losses['acc'] += loss_acc
        loss = sum(losses.values())

        loss.backward()

        if elasticity_requires_grad:
            elasticity_grad_norm = clip_grad_norm_(
                elasticity.parameters(),
                max_norm=cfg.train.elasticity_grad_max_norm,
                error_if_nonfinite=True)
            elasticity_optimizer.step()

        if plasticity_requires_grad:
            plasticity_grad_norm = clip_grad_norm_(
                plasticity.parameters(),
                max_norm=cfg.train.plasticity_grad_max_norm,
                error_if_nonfinite=True)
            plasticity_optimizer.step()

        msgs = [
            cfg.name,
            time.strftime('%H:%M:%S'),
            'epoch {:{width}d}/{}'.format(epoch + 1, cfg.train.num_epochs, width=len(str(cfg.train.num_epochs))),
            'teachers {}'.format(num_teachers)
        ]

        if elasticity_requires_grad:
            elasticity_lr = elasticity_optimizer.param_groups[0]['lr']
            msgs.extend([
                'e-lr {:.2e}'.format(elasticity_lr),
                'e-|grad| {:.4f}'.format(elasticity_grad_norm),
            ])

        if plasticity_requires_grad:
            plasticity_lr = plasticity_optimizer.param_groups[0]['lr']
            msgs.extend([
                'p-lr {:.2e}'.format(plasticity_lr),
                'p-|grad| {:.4f}'.format(plasticity_grad_norm),
            ])

        for loss_k, loss_v in losses.items():
            msgs.append('{} {:.4f}'.format(loss_k, loss_v.item()))
            writer.add_scalar('loss/{}'.format(loss_k), loss_v.item(), epoch + 1)

        msg = ','.join(msgs)
        tqdm.write('[{}]'.format(msg))

        torch.save({
            'elasticity': elasticity.state_dict(),
            'plasticity': plasticity.state_dict()
        }, ckpt_root / '{:04d}.pt'.format(epoch + 1))

        if elasticity_requires_grad:
            elasticity_lr = elasticity_optimizer.param_groups[0]['lr']
            writer.add_scalar('lr/elasticity', elasticity_lr, epoch + 1)
            writer.add_scalar('grad_norm/elasticity', elasticity_grad_norm, epoch + 1)
            elasticity_lr_scheduler.step()
        if plasticity_requires_grad:
            plasticity_lr = plasticity_optimizer.param_groups[0]['lr']
            writer.add_scalar('lr/plasticity', plasticity_lr, epoch + 1)
            writer.add_scalar('grad_norm/plasticity', plasticity_grad_norm, epoch + 1)
            plasticity_lr_scheduler.step()
        writer.add_scalar('teacher/lambda', teacher.curr_lambda, epoch + 1)
        writer.add_scalar('teacher/number', num_teachers, epoch + 1)
        writer.add_scalar('teacher/loss_factor', loss_factor, epoch + 1)
        teacher.step()

    writer.close()


if __name__ == '__main__':
    main()
