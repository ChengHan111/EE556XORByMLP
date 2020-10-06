import random
import numpy as np
random.seed(0)

def generate_zero():
    return random.uniform(0, 49) / 100

def generate_one():
    return random.uniform(50, 100) / 100


def generate_xor_XY(num_data_points):
    Xs, Ys = [], []
    for _ in range(num_data_points):
        # xor(0, 0) -> 0
        Xs.append([generate_zero(), generate_zero()]); Ys.append([0])
        # xor(1, 0) -> 1
        Xs.append([generate_one(), generate_zero()]); Ys.append([1])
        # xor(0, 1) -> 1
        Xs.append([generate_zero(), generate_one()]);
        Ys.append([1])
        # xor(1, 1) -> 0
        Xs.append([generate_one(), generate_one()]);
        Ys.append([0])
    return Xs, Ys
X, Y = generate_xor_XY(100)
X = np.array(X)
Y = np.array(Y)
for i, (x, y) in enumerate(zip(X, Y)):
    if i > 20:
        break
    print(x, [int(_x > 0.5) for _x in x], y)
