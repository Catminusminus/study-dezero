if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


import time
import dezero
import dezero.functions as F
from dezero import optimizers
from dezero import Dataloader
from dezero.models import MLP


max_epoch = 5
batch_size = 100

train_set = dezero.datasets.MNIST(train=True)
train_loader = dezero.Dataloader(train_set, batch_size)
model = MLP((1000, 10))
optimizer = optimizers.SGD().setup(model)

if dezero.cuda.gpu_enable:
    train_loader.to_gpu()
    model.to_gpu()

for epoch in range(max_epoch):
    start = time.time()
    sum_loss = 0

    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data) * len(t)

    elapsed_time = time.time() - start
    print(
        f"epoch: {epoch + 1}, loss: {sum_loss / len(train_set)}, time: {elapsed_time}[sec]"
    )
