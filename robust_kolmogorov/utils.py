from collections.abc import Iterable
from importlib import import_module
import os
from pathlib import Path
import random

from dotenv import load_dotenv
import numpy as np
import plotly.graph_objects as go
import torch


WANDB_CONFIG_PATH_ENV = "WANDB_CONFIG_PATHS"
WANDB_NAME_ENV = "WANDB_NAME"
WANDB_TAGS_ENV = "WANDB_TAGS"
WANDB_NOTES_ENV = "WANDB_NOTES"
WANDB_RUN_GROUP_ENV = "WANDB_RUN_GROUP"

CUDA_OOM_MSG = "CUDA out of memory"


def setup(overwrite_config=None):
    # load env from `.env` file
    load_dotenv()

    # automatically adapt wandb env variables based on the given configs
    overwrite_config = overwrite_config or {}
    overwrite_tags = [f"{k}={v}" for k, v in overwrite_config.items()]
    configs = os.environ.get(WANDB_CONFIG_PATH_ENV)
    config_names = [Path(p.strip()).stem for p in configs.split(",")] if configs else []

    if config_names or overwrite_tags:
        if WANDB_NAME_ENV not in os.environ:
            os.environ[WANDB_NAME_ENV] = " ".join(config_names + overwrite_tags)

        wandb_tags = (
            os.environ[WANDB_TAGS_ENV].split(",")
            if WANDB_TAGS_ENV in os.environ
            else []
        )
        os.environ[WANDB_TAGS_ENV] = ",".join(
            wandb_tags + config_names + overwrite_tags
        )

        wandb_notes = os.environ.get(WANDB_NOTES_ENV, "")
        if config_names:
            if WANDB_RUN_GROUP_ENV not in os.environ:
                os.environ[WANDB_RUN_GROUP_ENV] = " ".join(config_names[:-1])
            wandb_notes += f"\nConfigs: {configs}"
        if overwrite_tags:
            wandb_notes += f"\nOverwrite: {','.join(overwrite_tags)}"
        os.environ[WANDB_NOTES_ENV] = wandb_notes


def make_iterable(value):
    if isinstance(value, Iterable) and not isinstance(value, str):
        return value
    return [value]


def import_string(path):
    """Import a module path and return the attribute/class designated
    by the last name in the dotted path. Raise ImportError if the import failed."""
    try:
        module_path, class_name = path.rsplit(".", 1)
    except ValueError as err:
        raise ImportError(f"`{path}` does not look like a dotted module path.") from err

    module = import_module(module_path)

    try:
        return getattr(module, class_name)
    except AttributeError as err:
        raise ImportError(
            f"Module `{module_path}` does not define a `{class_name}` attribute/class."
        ) from err


def nest_dict(flat_dict):
    """Return nested dict by splitting the dotted keys."""
    nested_dict = {}
    for key, val in flat_dict.items():
        parent = nested_dict
        substrings = key.split(".")
        for part in substrings[:-1]:
            parent = parent.setdefault(part, {})
        parent[substrings[-1]] = val
    return nested_dict


def byte2mb(byte, ndigits=2):
    return round(byte / 1024**2, ndigits)


def check_mem(mem, mem_bound=None, rel_margin=0.0, device=None):
    # get cuda device memory
    device_mem_bound = float("inf")
    if device and device.type == "cuda":
        device_mem_bound = torch.cuda.get_device_properties(device).total_memory

    # compute overall bound
    mem_bound = (
        device_mem_bound if mem_bound is None else min(device_mem_bound, mem_bound)
    )
    # check
    mem = make_iterable(mem)
    if any(mem_bound < m * (1.0 + rel_margin) for m in mem):
        raise RuntimeError(
            f"{CUDA_OOM_MSG}. Tried to allocate a maximum of {byte2mb(max(mem))} MiB and "
            f"{byte2mb(rel_margin * max(mem))} MiB margin ({byte2mb(mem_bound)} MiB allowed)."
        )


def catch_oom(fn, *args, **kwargs):
    try:
        return None, fn(*args, **kwargs)
    except RuntimeError as err:
        # catch cuda out of memory errors
        if CUDA_OOM_MSG in str(err):
            return err, None
        raise


def max_mem_allocation(fn, *args, device=None, add_oom=1, empty_cache=True, **kwargs):
    if device.type == "cuda":
        if empty_cache:
            torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    err, output = catch_oom(fn, *args, **kwargs)
    if err:
        return torch.cuda.get_device_properties(device).total_memory + add_oom, output
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device), output
    return 0.0, output


def determinism(seed):
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_generators(n, device=None):
    seeds = random.sample(range(10 * 13, 10**18), n)
    generators = []
    for seed in seeds:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        generators.append(generator)
    return generators


def grad_fn(fn, *args, create_graph=True, **kwargs):
    def closure(x, t):
        # require gradient
        grad_flag = x.requires_grad
        x.requires_grad_(True)
        # calculate gradient w.r.t. x
        grad = torch.autograd.grad(
            fn(x, t, *args, **kwargs).sum(), x, create_graph=create_graph
        )[0]
        # revert to previous setting
        x.requires_grad_(grad_flag)
        return grad

    return closure


def pairwise_cos_similarity(x1, x2, dim=1, eps=1e-6):
    cos_similarity = torch.nn.CosineSimilarity(dim=dim, eps=eps)(
        x1.unsqueeze(-1), x2.T.unsqueeze(0)
    )
    row, col = torch.triu_indices(*cos_similarity.shape, device=x1.device)
    return cos_similarity[row, col]


def plot_density(named_values, bins=60, fig=None):
    fig = fig or go.Figure()
    min_value = (
        torch.stack([values.min() for values in named_values.values()]).min().item()
    )
    max_value = (
        torch.stack([values.max() for values in named_values.values()]).max().item()
    )
    delta = (max_value - min_value) / bins
    x = torch.linspace(min_value + delta / 2, max_value - delta / 2, bins, device="cpu")

    # disable deterministic algorithms for `torch.histc`
    # see https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    mode = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)

    for name, values in named_values.items():
        density = torch.histc(values, bins=bins, min=min_value, max=max_value) / (
            values.numel() * delta
        )
        fig.add_trace(go.Scatter(x=x, y=density.cpu(), mode="lines", name=name))

    # revert old mode
    torch.use_deterministic_algorithms(mode)
    return fig


def plot(named_values, fig=None):
    fig = fig or go.Figure()

    for name, (x, y) in named_values.items():
        fig.add_trace(
            go.Scatter(
                x=x.cpu().squeeze(), y=y.cpu().squeeze(), mode="lines", name=name
            )
        )
    return fig


class LpMetrics:
    suffixes = ("", "_rel")

    def __init__(self, p_values=1, eps=1.0):
        self.p_values = make_iterable(p_values)
        if not all(
            p == "infinity" or isinstance(p, (float, int)) for p in self.p_values
        ):
            raise ValueError("p can either be a number or 'infinity'.")

        self.eps = eps
        self.running = None
        self.zero()

    def zero(self):
        self.running = {
            (p, suffix): [] for p in self.p_values for suffix in LpMetrics.suffixes
        }

    @staticmethod
    def name(p, suffix, prefix=""):
        return f"{prefix}L{p}" + suffix

    def __call__(self, y, prediction, prefix=""):
        assert y.shape == prediction.shape
        metrics = {}
        norm_diff = torch.linalg.norm(prediction - y, dim=1)
        rel_abs_diff = norm_diff / (torch.linalg.norm(y, dim=1) + self.eps)

        for error, suffix in zip([norm_diff, rel_abs_diff], LpMetrics.suffixes):
            for p in self.p_values:
                if p == "infinity":
                    loss = error.max()
                else:
                    loss = (error**p).mean()

                self.running[(p, suffix)].append(loss.detach())
                metrics[LpMetrics.name(p, suffix, prefix=prefix)] = loss

        return metrics

    def result(self, prefix=""):
        metrics = {}

        for (p, suffix), values in self.running.items():
            if p == "infinity":
                metric = torch.stack(values).max()
            else:
                metric = torch.stack(values).mean() ** (1.0 / p)

            metrics[LpMetrics.name(p, suffix, prefix=prefix)] = metric.item()

        self.zero()
        return metrics


def zero_grad(model, set_to_none=False):
    """Similar to the `zero_grad` method of torch optimizers"""
    for p in model.parameters():
        if p.grad is not None:
            if set_to_none:
                p.grad = None
            else:
                if p.grad.grad_fn is not None:
                    p.grad.detach_()
                else:
                    p.grad.requires_grad_(False)
                p.grad.zero_()


def trainable_params(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])
