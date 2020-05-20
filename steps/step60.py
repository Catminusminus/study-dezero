if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


import numpy as np
import dezero
from dezero import Model, SeqDataLoader
import dezero.functions as F
import dezero.layers as L


class BetterRNN(Model):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.rnn = L.LSTM(hidden_size)
        self.rc = L.Linear(output_size)

    def reset_state(self):
        self.rnn.reset_state()
    
    def forward(self, x):
        y = self.rnn(x)
        y = self.rc(y)
        return y


if __name__ == "__main__":
    max_epoch = 300
    batch_size = 30
    hidden_size = 10
    bptt_length = 30

    train_set = dezero.datasets.SinCurve(train=True)
    dataloader = SeqDataLoader(train_set, batch_size=batch_size)
    seqlen = len(train_set)

    model = BetterRNN(hidden_size, 1)
    optimizer = dezero.optimizers.Adam().setup(model)

    for epoch in range(max_epoch):
        model.reset_state()
        loss, count = 0, 0

        for x, t in dataloader:
            y = model(x)
            loss += F.mean_square_error(y, t)
            count += 1

            if count % bptt_length == 0 or count == seqlen:

                model.cleargrads()
                loss.backward()
                loss.unchain_backward()
                optimizer.update()
        
        avg_loss = float(loss.data) / count
        print(f"| epoch {epoch + 1} | loss {avg_loss}")
