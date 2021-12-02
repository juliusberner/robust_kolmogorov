import math

import torch
from torch import nn


class Model(nn.Module):
    """
    Base class for different models.
    """

    def __init__(self, problem, grad_model=False, name="model"):
        super().__init__()
        self.problem = problem
        self.name = name

        # dims
        self.dim_in = self.problem.dim + 1
        self.dim_out = self.problem.dim if grad_model else 1

        # standardization
        intervals = [self.problem.interval, (0, self.problem.terminal_time)]
        self.mean = [sum(interval) / 2 for interval in intervals]
        self.std = [
            (interval[1] - interval[0]) / math.sqrt(12) for interval in intervals
        ]

    def standardize_and_flatten(self, x, t):
        return torch.cat(
            [(x - self.mean[0]) / self.std[0], (t - self.mean[1]) / self.std[1]], dim=1
        )


class FeedForward(Model):
    def __init__(
        self,
        arch,
        activation_factory,
        *args,
        normalization_factory=None,
        normalization_kwargs=None,
        name="FeedForward",
        **kwargs,
    ):
        super().__init__(*args, name=name, **kwargs)

        # affine linear layer
        bias = normalization_factory is None
        self.linear_layer = nn.ModuleList([nn.Linear(self.dim_in, arch[0], bias=bias)])
        self.linear_layer += [
            nn.Linear(arch[i], arch[i + 1], bias=bias) for i in range(len(arch) - 1)
        ]
        self.linear_layer.append(nn.Linear(arch[-1], self.dim_out))

        # activation function
        self.activation = activation_factory()

        # normalization layer
        if normalization_factory:
            normalization_kwargs = normalization_kwargs or {}
            self.norm_layer = nn.ModuleList(
                [
                    normalization_factory(num_features, **normalization_kwargs)
                    for num_features in self.arch
                ]
            )
        else:
            self.norm_layer = None

    def forward(self, x, t):
        tensor = self.standardize_and_flatten(x, t)

        for i, linear in enumerate(self.linear_layer[:-1]):
            tensor = self.activation(linear(tensor))
            if self.norm_layer is not None:
                tensor = self.norm_layer[i](tensor)

        return self.linear_layer[-1](tensor)


class DenseNet(Model):
    def __init__(
        self,
        arch,
        activation_factory,
        *args,
        name="DenseNet",
        **kwargs,
    ):
        super().__init__(*args, name=name, **kwargs)
        self.nn_dims = [self.dim_in] + arch + [self.dim_out]
        self.layers = nn.ModuleList(
            [
                nn.Linear(sum(self.nn_dims[: i + 1]), self.nn_dims[i + 1])
                for i in range(len(self.nn_dims) - 1)
            ]
        )
        self.activation = activation_factory()

    def forward(self, x, t):
        tensor = self.standardize_and_flatten(x, t)

        for i in range(len(self.nn_dims) - 1):
            if i == len(self.nn_dims) - 2:
                tensor = self.layers[i](tensor)
            else:
                tensor = torch.cat(
                    [tensor, self.activation(self.layers[i](tensor))], dim=1
                )
        return tensor


class LevelNet(Model):
    """
    Network module for a single level
    """

    def __init__(
        self,
        dim,
        level,
        activation_factory,
        *args,
        normalization_factory=None,
        normalization_kwargs=None,
        name="LevelNet",
        **kwargs,
    ):
        super().__init__(*args, name=name, **kwargs)

        self.level = level
        bias = normalization_factory is None
        self.dense_layers = nn.ModuleList([nn.Linear(self.dim_in, dim, bias=bias)])
        self.dense_layers += [
            nn.Linear(dim, dim, bias=bias) for _ in range(2**level - 1)
        ]
        self.dense_layers.append(nn.Linear(dim, self.dim_out))
        if normalization_factory is None:
            self.norm_layers = None
        else:
            normalization_kwargs = normalization_kwargs or {}
            self.norm_layers = nn.ModuleList(
                [
                    normalization_factory(dim, **normalization_kwargs)
                    for _ in range(2**level)
                ]
            )
        self.act = activation_factory()

    def forward(self, x, t, res_tensors=None):
        tensor = self.standardize_and_flatten(x, t)

        out_tensors = []
        tensor = self.dense_layers[0](tensor)
        for i, dense in enumerate(self.dense_layers[1:]):
            if self.norm_layers is not None:
                tensor = self.norm_layers[i](tensor)
            tensor = self.act(tensor)
            tensor = dense(tensor)
            if res_tensors:
                tensor = tensor + res_tensors[i]
            if i % 2 or self.level == 0:
                out_tensors.append(tensor)
        return out_tensors


class MultilevelNet(Model):
    """
    Multilevel net
    """

    def __init__(
        self,
        activation_factory,
        *args,
        factor=5,
        levels=4,
        normalization_factory=None,
        normalization_kwargs=None,
        name="MultilevelNet",
        **kwargs,
    ):
        super().__init__(*args, name=name, **kwargs)
        self.nets = nn.ModuleList(
            [
                LevelNet(
                    factor * self.dim_in,
                    level,
                    activation_factory,
                    *args,
                    normalization_factory=normalization_factory,
                    normalization_kwargs=normalization_kwargs,
                    name=f"Level {level}",
                    **kwargs,
                )
                for level in range(levels)
            ]
        )

    def forward(self, x, t):
        res_tensors = None
        for net in self.nets[::-1]:
            res_tensors = net(x, t, res_tensors)
        return res_tensors[-1]
