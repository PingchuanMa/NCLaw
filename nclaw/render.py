import math

import numpy as np


def get_camera():
    target = np.ones(3) * 0.5
    length = math.sqrt(2) * 1.7
    theta = 90 + 55
    y = 1.2
    x = math.cos(math.radians(theta)) * length
    z = math.sin(math.radians(theta)) * length
    origin = target + np.array([x, y, z])

    target[1] -= 0.15
    origin[1] -= 0.15

    return [
        origin.tolist(),
        target.tolist(),
        (0.0, 1.0, 0.0),
    ]


def get_slope_camera():
    target = np.array([0.5, 0.075, 0.3])
    target[1] -= 0.075
    # target[2] += 0.075
    length = math.sqrt(2) * 0.5
    theta = 90 + 60
    y = 0.3
    x = math.cos(math.radians(theta)) * length
    z = math.sin(math.radians(theta)) * length
    origin = target + np.array([x, y, z])

    return [
        origin.tolist(),
        target.tolist(),
        (0.0, 1.0, 0.0),
    ]
