import itertools
import unittest

from dotenv import load_dotenv
import torch

from robust_kolmogorov import utils, problems


class ProblemTestCase(unittest.TestCase):
    def __init__(
        self,
        *args,
        dim=2,
        batch_size=2,
        time_step=0.01,
        samples=8192,
        mc_rounds=512,
        atol=1e-2,
        rtol=1e-2,
        device=None,
        dtype=torch.float,
        seed=42,
        **kwargs,
    ):
        load_dotenv()
        self.dim = dim
        self.batch_size = batch_size
        self.time_step = time_step
        self.samples = samples
        self.mc_rounds = mc_rounds
        self.atol = atol
        self.rtol = rtol
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.seed = seed
        super().__init__(*args, **kwargs)

    def simulate(self, x, t, problem, mc_rounds, grad_fn=None):
        sde, stoch_integral = 0.0, 0.0
        for i in range(mc_rounds):
            sde_sim, stoch_integral_sim = problem.simulate(
                x,
                t,
                self.time_step,
                grad_fn=grad_fn,
            )
            self.assertFalse(sde_sim.requires_grad)
            self.assertFalse(stoch_integral_sim.requires_grad)
            self.assertTrue(stoch_integral_sim.shape == sde_sim.shape == t.shape)

            sde += sde_sim
            stoch_integral += stoch_integral_sim

        sde /= mc_rounds
        stoch_integral /= mc_rounds
        return sde, stoch_integral

    def assert_monte_carlo(self, problem, grad_fn=None, check_grad=True):
        print(f"--- {problem.name} ---")
        problem.setup()
        x, t = problem.sample(self.batch_size)
        grad_fn = grad_fn or problem.gradient
        if check_grad:
            grad_fn_ad = utils.grad_fn(problem.solution, create_graph=False)
            torch.testing.assert_close(
                grad_fn_ad(x, t), grad_fn(x, t), atol=self.atol, rtol=self.rtol
            )

        y = problem.solution(x, t)
        print(f"Solution:\n{y.flatten()}")
        self.assertTrue(x.shape == (self.batch_size, problem.dim))
        self.assertTrue(y.shape == t.shape == (self.batch_size, 1))

        sde, stoch_integral = self.simulate(
            x.repeat(self.samples, 1),
            t.repeat(self.samples, 1),
            problem,
            self.mc_rounds,
        )
        pred = sde.view(-1, *t.shape).mean(dim=0)
        print(f"MC SDE:\n{pred.flatten()}")
        self.assertFalse(stoch_integral.any())
        torch.testing.assert_close(pred, y, atol=self.atol, rtol=self.rtol)

        if grad_fn:
            sde, stoch_integral = self.simulate(
                x.repeat(self.samples, 1),
                t.repeat(self.samples, 1),
                problem,
                1,
                grad_fn=grad_fn,
            )
            pred = (sde - stoch_integral).view(-1, *t.shape).mean(dim=0)
            print(f"SDE - stoch. integral:\n{pred.flatten()}")
            torch.testing.assert_close(pred, y, atol=self.atol, rtol=self.rtol)

    def test_heat(self):
        for diffusivity, alpha in itertools.product([0.125, 1.5], [1.0, 1.5]):
            with self.subTest(diffusivity=diffusivity, alpha=alpha):
                utils.determinism(self.seed)
                problem = problems.Heat(
                    diffusivity=diffusivity,
                    alpha=alpha,
                    dim=self.dim,
                    device=self.device,
                    dtype=self.dtype,
                )
                self.assert_monte_carlo(problem)

    def test_heat_fourth_moment(self):
        utils.determinism(self.seed)
        problem = problems.HeatFourthMoment(
            dim=self.dim, device=self.device, dtype=self.dtype
        )
        self.assert_monte_carlo(problem)

    def test_heat_gaussian(self):
        for diffusivity in [0.125, 0.5]:
            with self.subTest(diffusivity=diffusivity):
                utils.determinism(self.seed)
                problem = problems.HeatGaussian(
                    diffusivity=diffusivity,
                    dim=self.dim,
                    device=self.device,
                    dtype=self.dtype,
                )
                self.assert_monte_carlo(problem)

    def test_linearized_hjb(self):
        for kappa in [0.1, 0.2]:
            with self.subTest(kappa=kappa):
                utils.determinism(self.seed)
                problem = problems.LinearizedHJB(
                    kappa=kappa, dim=self.dim, device=self.device, dtype=self.dtype
                )
                self.assert_monte_carlo(problem, check_grad=False)

    def test_black_scholes(self):
        for drift in [-0.05, -0.1]:
            with self.subTest(drift=drift):
                utils.determinism(self.seed)
                problem = problems.BlackScholes(
                    drift=drift, dim=self.dim, device=self.device, dtype=self.dtype
                )
                problem_coarse = problems.BlackScholes(
                    mc_samples=8,
                    drift=drift,
                    dim=self.dim,
                    device=self.device,
                    dtype=self.dtype,
                )
                problem_coarse.setup()
                self.assert_monte_carlo(
                    problem, grad_fn=problem_coarse.gradient, check_grad=False
                )
