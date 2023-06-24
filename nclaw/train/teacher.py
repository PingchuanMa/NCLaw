import math
import numpy as np
from omegaconf import DictConfig


class Teacher(object):

    def __init__(self, cfg: DictConfig) -> None:
        self.strategy: str = cfg.train.teacher.strategy
        if self.strategy in ['linear', 'cosine']:
            self.forward = self.uniform
        elif self.strategy == 'random':
            self.forward = self.random
        elif self.strategy == 'student':
            self.forward = self.student
        else:
            raise ValueError('invalid teacher strategy: {}'.format(self.strategy))
        self.num_epochs: int = cfg.train.num_epochs
        self.start_lambda: float = cfg.train.teacher.start_lambda
        self.end_lambda: float = cfg.train.teacher.end_lambda
        if self.strategy in ['linear', 'cosine']:
            self.end_lambda: float = min(self.end_lambda, cfg.sim.num_steps)

        self.last_epoch: int = 0
        self.curr_lambda: float = self.start_lambda

    def step(self) -> None:
        self.last_epoch += 1

        progress = self.last_epoch / self.num_epochs
        if self.strategy == 'cosine':
            progress = 0.5 * (math.cos(progress * math.pi) + 1.0)
        else:
            progress = 1 - progress

        self.curr_lambda = self.end_lambda + (self.start_lambda - self.end_lambda) * progress

    def uniform(self, num_steps: int) -> list[bool]:
        num_segments = math.ceil(num_steps / self.curr_lambda)
        trajectory = np.floor(np.linspace(0, num_segments, num_steps, endpoint=False)).astype(int).tolist()
        prev_val = -1
        for i, val in enumerate(trajectory):
            trajectory[i] = val != prev_val
            prev_val = val
        return trajectory

    def random(self, num_steps: int) -> list[bool]:
        trajecotry = (np.random.random(num_steps) < self.curr_lambda).tolist()
        return trajecotry

    def student(self, num_steps: int) -> list[bool]:
        trajecotry = [False for step in range(num_steps)]
        return trajecotry

    def __call__(self, num_steps: int) -> list[bool]:
        trajecotry = self.forward(num_steps)
        trajecotry[0] = True
        return trajecotry

    def num_teachers(self, num_steps: int) -> int:
        teachers = self(num_steps)
        return sum([int(t) for t in teachers])

    def loss_factor(self, num_steps: int) -> float:
        if self.strategy in ['linear', 'cosine']:
            return math.ceil(num_steps / self.curr_lambda) / num_steps
        return 1.0


if __name__ == '__main__':
    class TestTeacher(object):
        def __init__(self) -> None:
            self.curr_lambda = 600
    a = TestTeacher()

    teachers = Teacher.uniform(a, 1000)
    print(np.array(teachers))
    print(sum([int(t) for t in teachers]))
