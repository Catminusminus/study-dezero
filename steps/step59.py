if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


from dezero import Model
import dezero.functions as F
import dezero.layers as L
import dezero


class SimpleRNN(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.rnn = L.RNN(hidden_size)
        self.fc = L.Linear(out_size)

    def reset_state(self):
        self.rnn.reset_state()

    def forward(self, x):
        h = self.rnn(x)
        y = self.fc(h)
        return y


if __name__ == "__main__":
    max_epoch = 100
    hidden_size = 100
    bptt_length = 30

    train_set = dezero.datasets.SinCurve(train=True)
    seqlen = len(train_set)

    model = SimpleRNN(hidden_size, 1)
    optimizer = dezero.optimizers.Adam().setup(model)

    for epoch in range(max_epoch):
        model.reset_state()
        loss, count = 0, 0

        for x, t in train_set:
            x = x.reshape(1, 1)
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