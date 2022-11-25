import syft as sy
import torch as th
from syft import WebsocketClientWorker
from torch import nn

hook = sy.TorchHook(th)


class Net(th.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = Net()
mock_data = th.zeros(1, 2)
traced_model = th.jit.trace(model, mock_data)
print(type(traced_model))


@th.jit.script
def loss_fn(target, pred):
    return ((target.view(pred.shape).float() - pred.float()) ** 2).mean()


print(type(loss_fn))
optimizer = "SGD"
batch_size = 4
optimizer_args = {'lr': 0.1, "weight_decay": 0.01}
epochs = 1
max_nr_batches = -1
shuffle = True

train_config = sy.TrainConfig(model=traced_model, loss_fn=loss_fn, optimizer=optimizer, batch_size=batch_size,
                              optimizer_args=optimizer_args, epochs=epochs, shuffle=shuffle)

alice = WebsocketClientWorker(id="alice", port=8777, hook=hook, host="localhost", verbose=True)
train_config.send(alice)
data = th.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]], requires_grad=True)
target = th.tensor([[1.0], [1.0], [0.0], [0.0]], requires_grad=False)

print("Evaluation before training")
pred = model(data)
loss = loss_fn(target=target, pred=pred)
print(f"Loss: {loss}")
print(f"Target: {target}")
print(f"Pred: {pred}")

for epoch in range(300):
    loss = alice.fit(dataset_key="xor")
    if epoch % 50 == 0:
        print("-" * 50)
        print(f"Iteration {epoch}: alice loss: {loss}")

new_model = train_config.model_ptr.get()

print("Evaluation after training")
pred = new_model(data)
loss = loss_fn(target=target, pred=pred)
print(f"Loss: {loss}")
print(f"Target: {target}")
print(f"Pred: {pred}")

if __name__ == '__main__':
    print("hello")
