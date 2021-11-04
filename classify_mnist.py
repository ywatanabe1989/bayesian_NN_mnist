#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2021-11-05 02:53:50 (ywatanabe)"

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
import torch.nn.functional as F

# from pyro.nn.module import to_pyro_module_
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.nn import PyroModule, PyroParam  # , PyroSample
from pyro.optim import Adam
from tqdm import tqdm

# import pyro.poutine as poutine
# from torch.distributions import constraints


# For more details, please see https://pyro.ai/examples/modules.html

smoke_test = "CI" in os.environ
assert pyro.__version__.startswith("1.7.0")


class BayesianLinear(PyroModule):
    def __init__(self, in_size, out_size):
        super().__init__()

        self.register_buffer("bias", torch.ones((out_size,)))
        self.weight = PyroParam(torch.randn((in_size, out_size)))

    def forward(self, input):
        return self.bias + input @ self.weight  # this line samples bias and weight


class BayesianPerceptron(PyroModule):
    def __init__(self, in_size, hidden_size, out_size, *args, **kwargs):
        assert len(args) == 0
        assert len(kwargs) == 0

        super().__init__()

        self.fc1 = BayesianLinear(in_size, hidden_size)
        self.fc2 = BayesianLinear(hidden_size, out_size)

        self.dropout_layer = nn.Dropout(0.5)
        self.act_func = nn.Sigmoid()  # nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout_layer(x)
        x = self.act_func(x)
        x = self.fc2(x)
        return x


class Model(PyroModule):
    def __init__(self, model, *args, **kwargs):
        super().__init__()
        self.model = model(*args, **kwargs)  # this is a PyroModule

    def forward(self, input, output=None):
        logits = self.model(input)  # this samples linear.bias and linear.weight
        conf = F.gumbel_softmax(logits, tau=1, hard=True, dim=-1)  # hard=False

        with pyro.plate("instances", len(input), device=DEVICE):
            return pyro.sample(
                "obs",
                dist.Normal(conf, 0.1 * torch.ones_like(conf)).to_event(1),
                obs=output,
            )


if __name__ == "__main__":
    import torchvision

    ################################################################################
    ## Fixes random seeds
    ################################################################################
    SEED = 42
    ## os
    os.environ["PYTHONHASHSEED"] = str(SEED)
    ## torch
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    ## numpy
    np.random.seed(SEED)
    ## pyro
    pyro.set_rng_seed(SEED)

    ################################################################################
    ## Parameters
    ################################################################################
    DEVICE = "cuda"
    BATCH_SIZE = 5120
    IN_SIZE = 28 * 28  # MNIST
    HIDDEN_SIZE = 1000
    OUT_SIZE = 10
    MAX_EPOCHS = 100

    ################################################################################
    ## Prepares demo data
    ################################################################################
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
        ]
    )

    dl_tra = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            "/tmp/mnist-data/",
            train=True,
            download=True,
            transform=transform,
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    dl_tes = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            "/tmp/mnist-data/", train=False, transform=transform
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    ################################################################################
    ## Model
    ################################################################################
    pyro.clear_param_store()

    model = Model(BayesianPerceptron, IN_SIZE, HIDDEN_SIZE, OUT_SIZE).cuda()
    guide = AutoNormal(model).cuda()
    svi = SVI(model, guide, Adam({"lr": 1e-2}), Trace_ELBO())

    ################################################################################
    ## Training
    ################################################################################
    losses_tra = []
    accs_tra = []
    for epoch in tqdm(range(MAX_EPOCHS)):
        for i_batch, batch in enumerate(dl_tra):
            Xb_tra, Tb_tra = batch
            Xb_tra, Tb_tra = Xb_tra.to(DEVICE), Tb_tra.to(DEVICE)
            Xb_tra = Xb_tra.reshape(len(Xb_tra), -1)
            Tb_tra_onehot = F.one_hot(Tb_tra, num_classes=10)

            loss_tra = svi.step(Xb_tra, Tb_tra_onehot) / Tb_tra_onehot.numel()
            losses_tra.append(loss_tra)

            ## acc
            pred_conf_tra = model(Xb_tra)
            pred_conf_max_tra, pred_class_tra = pred_conf_tra.max(dim=-1)
            acc_tra = (pred_class_tra == Tb_tra).float().mean()
            accs_tra.append(acc_tra.item())

            if i_batch % 10 == 0:
                print(
                    f"i_batch {i_batch}; loss = {loss_tra:0.4g}; acc = {acc_tra:0.4g}"
                )

    ################################################################################
    ## Prediction
    ################################################################################
    T_all_tes = []
    pred_classes_tes = []
    losses_tes = []
    model.eval()
    for i_batch, batch in enumerate(dl_tes):
        Xb_tes, Tb_tes = batch
        Xb_tes, Tb_tes = Xb_tes.cuda(), Tb_tes.cuda()
        Xb_tes = Xb_tes.reshape(len(Xb_tes), -1)
        Tb_tes_onehot = F.one_hot(Tb_tes, num_classes=10)

        pred_conf_tes = model(Xb_tes)
        loss_tes = svi.evaluate_loss(Xb_tes, Tb_tes_onehot) / Tb_tes_onehot.numel()
        pred_conf_max_tes, pred_class_tes = pred_conf_tes.max(dim=-1)

        losses_tes.append(loss_tes)
        T_all_tes.append(Tb_tes)
        pred_classes_tes.append(pred_class_tes)

    acc_tes = (
        (torch.hstack(T_all_tes) == torch.hstack(pred_classes_tes))
        .float()
        .mean()
        .item()
    )
    loss_tes = np.mean(losses_tes)

    ################################################################################
    ## Plots the results
    ################################################################################
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(losses_tra, label="Training", color="blue")
    axes[0].scatter(len(losses_tra) - 1, loss_tes, label="Test", color="red")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(accs_tra, label="Training", color="blue")
    axes[1].set_ylabel("Accuracy")
    axes[1].scatter(len(accs_tra) - 1, acc_tes, label="Test", color="red")
    axes[1].set_ylim([0, 1])
    axes[1].legend()

    fig.supxlabel("Iteration #")
    fig.suptitle(f"Training | Max epochs: {MAX_EPOCHS} | Batch size: {BATCH_SIZE}")
    fig.tight_layout()
    # fig.show()
    fig.savefig("./classify_mnist_learning_curve.png")

    ## EOF
