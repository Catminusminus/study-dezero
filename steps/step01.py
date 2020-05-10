import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data


if __name__ == "__main__":
    data = np.array(1.0)
    x = Variable(data)

    print(x.data)

    data = np.array(2.0)
    x = Variable(data)

    print(x.data)
