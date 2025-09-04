# fl_project/client_app.py
"""fl-project: Flower ClientApp for Enron TF-IDF + Logistic (PyTorch)."""

import os
import numpy as np
import torch

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from fl_project.task import Net, get_weights, load_data, set_weights, test, train


def _get_f(name: str, default: float) -> float:
    v = os.environ.get(name, "").strip()
    try:
        return float(v) if v else default
    except Exception:
        return default


class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def fit(self, parameters, config):
        # Reçoit per-round config du serveur
        lr = float(config.get("lr", _get_f("LR", 1e-3)))
        clip_c = float(config.get("clip-c", _get_f("CLIP_C", 0.0)))
        set_weights(self.net, parameters)
        train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            self.device,
            lr=lr,
            clip_c=clip_c if clip_c > 0 else 0.0,
        )
        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {"train_loss": float(train_loss), "lr": float(lr), "clip_c": float(clip_c)},
        )

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return float(loss), len(self.valloader.dataset), {"val_acc": float(accuracy)}


def client_fn(context: Context):
    # Le template passe: partition-id & num-partitions via node_config
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])

    # load_data renvoie loaders + input_dim
    trainloader, valloader, input_dim = load_data(partition_id, num_partitions)

    # Construire le modèle avec le bon input_dim
    net = Net(input_dim=input_dim)

    # local-epochs fourni par run_config (sinon défaut 1)
    local_epochs = int(context.run_config.get("local-epochs", 1))

    return FlowerClient(net, trainloader, valloader, local_epochs).to_client()


# ClientApp
app = ClientApp(client_fn)
