import argparse

import syft as sy
import torch as th
from syft import WebsocketServerWorker

parser = argparse.ArgumentParser()
parser.add_argument("--port", "-p", type=int)
parser.add_argument("--host", type=str, default="localhost")
parser.add_argument("--id", type=str)


def main(**kwargs):
    # Use a breakpoint in the code line below to debug your script.
    print(kwargs)

    worker = WebsocketServerWorker(**kwargs)
    data = th.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]], requires_grad=True)
    target = th.tensor([[1.0], [1.0], [0.0], [0.0]], requires_grad=False)
    dataset = sy.BaseDataset(data, target)
    worker.add_dataset(dataset, key="xor")

    worker.start()
    return worker


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    hook = sy.TorchHook(th)
    args = parser.parse_args()
    kwargs = {
        "id": args.id,
        "host": args.host,
        "port": args.port,
        "hook": hook,
        "verbose": True
    }

    main(**kwargs)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/