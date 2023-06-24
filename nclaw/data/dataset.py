from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch.utils.data import Dataset


class MPMDataset(Dataset):

    def __init__(self, root: Union[str, Path], device: torch.device) -> None:
        super().__init__()

        with torch.no_grad():
            self.root = Path(root).resolve() / 'state'
            self.traj = list(sorted(self.root.glob('*.pt')))
            states = [torch.load(p, map_location=device) for p in self.traj]

            self.xs = torch.stack([state['x'] for state in states], dim=0)
            self.vs = torch.stack([state['v'] for state in states], dim=0)
            self.Cs = torch.stack([state['C'] for state in states], dim=0)
            self.Fs = torch.stack([state['F'] for state in states], dim=0)
            self.stresss = torch.stack([state['stress'] for state in states], dim=0)

    def __len__(self) -> int:
        return self.xs.size(0)

    def __getitem__(self, index):
        return self.xs[index], self.vs[index], self.Cs[index], self.Fs[index], self.stresss[index]
