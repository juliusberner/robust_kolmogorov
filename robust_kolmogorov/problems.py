from abc import ABC, abstractmethod

import numpy as np
from scipy.linalg import solve_banded
import torch

from robust_kolmogorov import utils


class Problem(ABC):
    """
    Base class for different problems.
    """

    def __init__(
        self,
        dim=1,
        interval=(-0.5, 0.5),
        terminal_time=1.0,
        device="cpu",
        dtype=torch.float,
        name="problem",
    ):
        super().__init__()
        self.dim = dim
        self.interval = interval
        self.terminal_time = terminal_time
        self.name = name
        self.device = device
        self.dtype = dtype

    def sample(self, batch_size=None, generator=None):
        x_dim = self.dim if batch_size is None else (batch_size, self.dim)
        t_dim = 1 if batch_size is None else (batch_size, 1)
        x = torch.empty(*x_dim, device=self.device, dtype=self.dtype).uniform_(
            *self.interval, generator=generator
        )
        t = torch.empty(*t_dim, device=self.device, dtype=self.dtype).uniform_(
            0.0, self.terminal_time, generator=generator
        )
        return x, t

    def simulate(
        self,
        x,
        t,
        time_step,
        grad_fn=None,
        weights=None,
        detach_integral=False,
        generator=None,
    ):
        # initialize
        time_step = torch.tensor(time_step, device=self.device, dtype=self.dtype)
        max_steps = int(torch.ceil((self.terminal_time - t).max() / time_step).item())
        stoch_integral = torch.zeros(t.shape, device=self.device, dtype=self.dtype)

        # Euler-Maruyama
        for _ in range(max_steps):
            time_delta = torch.nn.functional.relu(
                torch.minimum(time_step, self.terminal_time - t)
            )

            diffusion_increment = (
                self.diffusion(x, t)
                @ torch.randn(
                    *x.shape,
                    1,
                    device=self.device,
                    dtype=self.dtype,
                    generator=generator,
                )
            ).squeeze(-1) * torch.sqrt(time_delta)

            if grad_fn:
                stoch_integral_increment = (grad_fn(x, t) * diffusion_increment).sum(
                    1, keepdim=True
                )
                if weights is not None:
                    partial_loss = 2 * (weights * stoch_integral_increment).mean()
                    partial_loss.backward()
                if detach_integral:
                    stoch_integral_increment = stoch_integral_increment.detach()

                stoch_integral += stoch_integral_increment

            x = x + self.drift(x, t) * time_delta + diffusion_increment
            t = t + time_delta

        return self.terminal_cond(x), stoch_integral

    def setup(self):
        pass

    @abstractmethod
    def terminal_cond(self, x):
        pass

    @abstractmethod
    def drift(self, x, t):
        pass

    @abstractmethod
    def diffusion(self, x, t):
        pass

    @abstractmethod
    def solution(self, x, t, generator=None):
        pass

    @abstractmethod
    def gradient(self, x, t, generator=None):
        pass


class Heat(Problem):
    def __init__(
        self,
        name="Heat equation (paraboloid)",
        diffusivity=0.125,
        alpha=1.0,
        *args,
        **kwargs,
    ):
        super().__init__(name=name, *args, **kwargs)
        self.factor = torch.tensor(
            2.0 * diffusivity, device=self.device, dtype=self.dtype
        )
        self.alpha = torch.tensor(alpha, device=self.device, dtype=self.dtype)
        self.diffusion_tensor = torch.sqrt(self.factor) * torch.eye(
            self.dim, device=self.device, dtype=self.dtype
        )
        self.drift_tensor = torch.tensor(0.0, device=self.device, dtype=self.dtype)

    def drift(self, x, t):
        return self.drift_tensor

    def diffusion(self, x, t):
        return self.diffusion_tensor

    def terminal_cond(self, x):
        return self.alpha * (x**2).sum(-1, keepdims=True)

    def solution(self, x, t, generator=None):
        return self.terminal_cond(x) + self.alpha * self.factor * self.dim * (
            self.terminal_time - t
        )

    def gradient(self, x, t, generator=None):
        return 2 * self.alpha * x


class HeatFourthMoment(Problem):
    def __init__(self, name="Heat equation (fourth moment)", *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.diffusion_tensor = torch.eye(
            self.dim, device=self.device, dtype=self.dtype
        )
        self.drift_tensor = torch.tensor(0.0, device=self.device, dtype=self.dtype)

    def drift(self, x, t):
        return self.drift_tensor

    def diffusion(self, x, t):
        return self.diffusion_tensor

    def terminal_cond(self, x):
        return (x**2).sum(-1, keepdims=True) ** 2

    def solution(self, x, t, generator=None):
        return (
            self.terminal_cond(x)
            + 4 * (x**2).sum(-1, keepdims=True) * (self.terminal_time - t)
            + 2 * (x**2).sum(-1, keepdims=True) * (self.terminal_time - t) * self.dim
            + 2 * (self.terminal_time - t) ** 2 * self.dim
            + (self.terminal_time - t) ** 2 * self.dim**2
        )

    def gradient(self, x, t, generator=None):
        return (
            4 * x * (x**2).sum(-1, keepdims=True)
            + 8 * x * (self.terminal_time - t)
            + 4 * x * (self.terminal_time - t) * self.dim
        )


class HeatGaussian(Problem):
    def __init__(
        self, name="Heat equation (Gaussian)", diffusivity=0.125, *args, **kwargs
    ):
        super().__init__(name=name, *args, **kwargs)
        self.factor = torch.tensor(
            2.0 * diffusivity, device=self.device, dtype=self.dtype
        )
        self.diffusion_tensor = torch.sqrt(self.factor) * torch.eye(
            self.dim, device=self.device, dtype=self.dtype
        )
        self.drift_tensor = torch.tensor(0.0, device=self.device, dtype=self.dtype)

    def drift(self, x, t):
        return self.drift_tensor

    def diffusion(self, x, t):
        return self.diffusion_tensor

    def terminal_cond(self, x):
        return torch.exp(-(x**2).sum(-1, keepdims=True))

    def solution(self, x, t, generator=None):
        denominator = 1.0 + 2.0 * (self.terminal_time - t) * self.factor
        return torch.exp(
            -(x**2).sum(-1, keepdims=True) / denominator
        ) / denominator ** (self.dim / 2.0)

    def gradient(self, x, t, generator=None):
        return (
            -2.0
            * x
            * self.solution(x, t, generator=generator)
            / (1.0 + 2.0 * (self.terminal_time - t) * self.factor)
        )


class LinearizedHJB(Problem):
    def __init__(
        self,
        *args,
        kappa=0.1,
        eta=0.04,
        interval=(-1.5, 1.5),
        delta_t_reference=0.005,
        xb_reference=2.5,
        nx_reference=1000,
        name="Linearized HJB",
        **kwargs,
    ):
        super().__init__(
            *args,
            name=name,
            interval=interval,
            **kwargs,
        )
        self.diffusion_tensor = torch.eye(
            self.dim, device=self.device, dtype=self.dtype
        )
        self.kappa = kappa
        self.eta = eta

        self.xb = xb_reference  # range of x is [-xb, xb]
        self.nx = nx_reference  # number of discretization points
        self.dx = 2.0 * self.xb / self.nx
        self.dt = delta_t_reference

        # will be initialized in setup
        self.discr_control = None
        self.discr_solution = None
        self.discr_derivative = None

    def drift(self, x, t):
        return -4.0 * self.kappa * x * (x**2 - 1)

    def potential(self, x):
        return self.kappa * (x**2 - 1) ** 2

    def diffusion(self, x, t):
        return self.diffusion_tensor

    def terminal_cond(self, x):
        return torch.exp(-(self.eta * (x - 1) ** 2).sum(-1, keepdims=True))

    def setup(self):
        xvec = np.linspace(-self.xb, self.xb, self.nx, endpoint=False)

        # assemble D^{-1} L D assuming Neumann boundary conditions
        mat = np.zeros([self.nx, self.nx])
        for i in range(self.nx):
            x = -self.xb + (i + 0.5) * self.dx
            if i > 0:
                x0 = -self.xb + (i - 0.5) * self.dx
                x1 = -self.xb + i * self.dx
                mat[i, i - 1] = (
                    -np.exp(
                        self.potential(x0) + self.potential(x) - 2 * self.potential(x1)
                    )
                    / self.dx**2
                )
                mat[i, i] = (
                    np.exp(2 * (self.potential(x) - self.potential(x1))) / self.dx**2
                )
            if i < self.nx - 1:
                x0 = -self.xb + (i + 1.5) * self.dx
                x1 = -self.xb + (i + 1) * self.dx
                mat[i, i + 1] = (
                    -np.exp(
                        self.potential(x0) + self.potential(x) - 2 * self.potential(x1)
                    )
                    / self.dx**2
                )
                mat[i, i] = (
                    mat[i, i]
                    + np.exp(2 * (self.potential(x) - self.potential(x1)))
                    / self.dx**2
                )

        mat = -mat / 2

        # diagonal of matrix D
        diag = np.exp(self.potential(xvec))

        # discretized solution
        nt = int(self.terminal_time / self.dt)
        band = -self.dt * np.vstack(
            [
                np.append(0.0, mat.diagonal(offset=1)),
                mat.diagonal() - nt / self.terminal_time,
                np.append(mat.diagonal(offset=-1), 0.0),
            ]
        )

        solution = np.empty([nt + 1, self.nx])
        solution[nt, :] = (
            self.terminal_cond(torch.from_numpy(xvec).unsqueeze(-1)).squeeze().numpy()
        )

        for n in range(nt - 1, -1, -1):
            solution[n, :] = diag * solve_banded(
                [1, 1], band, solution[n + 1, :] / diag
            )

        # discretized control
        control = np.diff(np.log(solution), axis=1) / self.dx

        # store the tensors on cpu to not use cuda memory during training
        self.discr_control = torch.tensor(control, dtype=self.dtype, device="cpu").t()
        self.discr_solution = torch.tensor(solution, dtype=self.dtype, device="cpu").t()

    def indices(self, x, t):
        i_x = torch.floor(
            (torch.clip(x, -self.xb, self.xb - 2 * self.dx) + self.xb) / self.dx
        ).long()
        i_t = torch.floor(t / self.dt).long()
        return i_x.cpu(), i_t.cpu()

    def optimal_control(self, x, t):
        return self.discr_control[self.indices(x, t)].to(device=self.device)

    def solution(self, x, t, generator=None):
        return torch.prod(
            self.discr_solution[self.indices(x, t)].to(device=self.device),
            dim=1,
            keepdim=True,
        )

    def gradient(self, x, t, generator=None):
        return self.optimal_control(x, t) * self.solution(x, t)


class BlackScholes(Problem):
    def __init__(
        self,
        *args,
        interval=(4.5, 5.5),
        strike_price=5.5,
        drift=-0.05,
        mc_samples=65536,
        mc_split_size=8,
        name="Black-Scholes model",
        **kwargs,
    ):
        super().__init__(*args, name=name, interval=interval, **kwargs)
        self.drift_tensor = torch.tensor(drift, device=self.device, dtype=self.dtype)
        self.strike_price_tensor = torch.tensor(
            strike_price, device=self.device, dtype=self.dtype
        )
        self.mc_samples = mc_samples
        self.mc_split_size = mc_split_size

        # will be initialized in setup
        self.diffusion_tensor = None
        self.diffusion_norms = None

    def setup(self):
        sigma = torch.linalg.cholesky(
            0.5 * torch.ones(self.dim, self.dim, device=self.device, dtype=self.dtype)
            + 0.5 * torch.eye(self.dim, device=self.device, dtype=self.dtype)
        )
        beta = 0.1 + torch.arange(
            1, self.dim + 1, device=self.device, dtype=self.dtype
        ) / (2 * self.dim)

        self.diffusion_tensor = sigma * beta.unsqueeze(-1)
        self.diffusion_norms = (self.diffusion_tensor**2).sum(dim=1)

    def drift(self, x, t):
        return self.drift_tensor * x

    def diffusion(self, x, t):
        return self.diffusion_tensor * x.unsqueeze(-1)

    def terminal_cond(self, x):
        return torch.nn.functional.relu(
            self.strike_price_tensor - x.min(-1, keepdims=True)[0]
        )

    def sde(self, x, t, generator=None):
        time_delta = self.terminal_time - t
        diffusion_increment = (
            self.diffusion_tensor
            @ torch.randn(
                *x.shape, 1, device=self.device, dtype=self.dtype, generator=generator
            )
        ).squeeze(-1) * torch.sqrt(time_delta)
        return self.terminal_cond(
            x
            * torch.exp(
                (self.drift_tensor - 0.5 * self.diffusion_norms) * time_delta
                + diffusion_increment
            )
        )

    def solution(self, x, t, generator=None):
        y = []
        for x_values, t_values in zip(
            x.split(self.mc_split_size), t.split(self.mc_split_size)
        ):
            y.append(
                self.sde(
                    x_values.expand(self.mc_samples, *x_values.shape),
                    t_values.expand(self.mc_samples, *t_values.shape),
                    generator=generator,
                ).mean(dim=0)
            )
        return torch.cat(y)

    def gradient(self, x, t, generator=None):
        return utils.grad_fn(self.solution, create_graph=False, generator=generator)(
            x, t
        )
