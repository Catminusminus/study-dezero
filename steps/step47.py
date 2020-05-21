if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


import numpy as np
from dezero import Variable, as_variable
from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP


def softmax1d(x):
    x = as_variable(x)
    y = F.exp(x)
    sum_y = F.sum(y)
    return y / sum_y


if __name__ == "__main__":
    model = MLP((10, 3))

    x = np.array([[0.2, -0.4]])
    y = model(x)
    print(y)
    p = softmax1d(y)
    print(p)
