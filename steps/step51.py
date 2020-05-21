if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


import dezero
import numpy as np
from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP


def f(x):
    x = x.flatten()
    x = x.astype(np.float32)
    x /= 255.0
    return x


train_set = dezero.datasets.MNIST(train=True, transform=f)
test_set = dezero.datasets.MNIST(train=False, transform=f)

max_epoch = 5
batch_size = 100
hidden_size = 1000

train_loader = dezero.Dataloader(train_set, batch_size)
test_loader = dezero.Dataloader(test_set, batch_size, shuffle=False)

model = MLP((hidden_size, hidden_size, 10), activation=F.relu)
optimizer = optimizers.SGD().setup(model)

for epoch in range(max_epoch):
    sum_loss, sum_acc = 0, 0
    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy_simple(y, t)
        acc = F.accuracy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)

    print(f"epoch: {epoch + 1}")
    print(f"train loss: {sum_loss / len(train_set)}, acc: {sum_acc / len(train_set)}")

    sum_loss, sum_acc = 0, 0
    with dezero.no_grad():
        for x, t in test_loader:
            y = model(x)
            loss = F.softmax_cross_entropy_simple(y, t)
            acc = F.accuracy(y, t)
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)

    print(f"test loss: {sum_loss / len(test_set)}, acc: {sum_acc / len(test_set)}")
