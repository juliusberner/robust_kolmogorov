from functools import partial
import os
import time

import torch
import wandb

from robust_kolmogorov import utils


class Solver:
    def __init__(
        self,
        problem_factory,
        model_factory,
        optimizer_factory,
        batch_size,
        time_step,
        loss_mode,
        train_time_h=None,
        train_steps=None,
        sample_repetitions=1,
        grad_model_factory=None,
        milestones=None,
        milestone_metric="step",
        check_train=False,
        check_test=False,
        lr=None,
        optimizer_kwargs=None,
        problem_kwargs=None,
        model_kwargs=None,
        grad_model_kwargs=None,
        test_freq=None,
        evaluator_kwargs=None,
        device=None,
        cuda_max_mem_mb=None,
        cuda_max_mem_train_mb=None,
        seed=None,
        dtype=torch.float,
        visualize=None,
        save_freq=None,
    ):

        # data type, device, and cuda memory
        self.dtype = dtype
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        total_memory = 0
        if self.device.type == "cuda":
            if self.device.index is None:
                self.device = torch.device(
                    self.device.type, torch.cuda.current_device()
                )
            total_memory = torch.cuda.get_device_properties(self.device).total_memory
            if cuda_max_mem_mb:
                fraction = min(cuda_max_mem_mb * 1024**2 / total_memory, 1.0)
                torch.cuda.set_per_process_memory_fraction(fraction, self.device)
        self.max_mem_train = (
            cuda_max_mem_train_mb * 1024**2 if cuda_max_mem_train_mb else None
        )

        wandb.run.summary["device"] = str(self.device)
        wandb.run.summary["total_cuda_mem"] = utils.byte2mb(total_memory)
        wandb.run.summary["cuda_device_count"] = torch.cuda.device_count()

        # seed and random number generators (to allow parts of the config to be changed w/o effect)
        if seed:
            utils.determinism(seed)
        (
            self.train_gen,
            self.approx_gen,
            self.plot_gen,
            self.stats_gen,
        ) = utils.get_generators(4, device=self.device)

        # milestones
        self.milestones = utils.make_iterable(milestones or [])
        if not (
            (milestone_metric == "step" and train_steps)
            or (milestone_metric == "time" and train_time_h)
        ):
            raise ValueError(
                "`milestone_metric` must be 'step' or 'time' and "
                "the corresponding argument (`train_steps` or `train_time_h`) must be defined."
            )
        self.milestone_metric = milestone_metric
        self.milestone_idx = 0
        self.check_train = check_train
        self.check_test = check_test

        # problem
        problem_kwargs = problem_kwargs or {}
        self.problem = problem_factory(
            device=self.device, dtype=self.dtype, **problem_kwargs
        )

        # training
        self.time_count = 0.0
        self.step_count = 0
        self.sample_count = 0
        self.train_time = float("inf") if train_time_h is None else train_time_h * 3600
        self.train_steps = float("inf") if train_steps is None else train_steps
        self.batch_size = batch_size
        self.loss_mode = loss_mode
        self.time_step = time_step
        self.sample_repetitions = sample_repetitions

        # models
        self.save_freq = save_freq
        self.save_count = 1
        model_kwargs = model_kwargs or {}
        self.model = model_factory(problem=self.problem, **model_kwargs)
        self.models = [self.model]
        wandb.run.summary["trainable_params"] = utils.trainable_params(self.model)

        self.grad_model = None
        if any(
            mode in ["model", "model_eff"]
            for mode in utils.make_iterable(self.loss_mode)
        ):
            grad_model_kwargs = grad_model_kwargs or {}
            self.grad_model = grad_model_factory(
                problem=self.problem, grad_model=True, **grad_model_kwargs
            )
            self.models.append(self.grad_model)
            wandb.run.summary["trainable_params_grad_model"] = utils.trainable_params(
                self.grad_model
            )

        for model in self.models:
            model.to(device=self.problem.device, dtype=self.problem.dtype)

        # test and visualization
        self.test_freq = test_freq
        self.test_count = 1
        self.visualize = visualize or []
        if "model" in self.visualize:
            wandb.watch(self.models, log="all", log_freq=1000, log_graph=True)

        evaluator_kwargs = evaluator_kwargs or {}
        self.evaluator = Evaluator(
            self.problem,
            self.model,
            grad_model=self.grad_model,
            **evaluator_kwargs,
        )

        # optimizer
        self.lr = lr
        optimizer_kwargs = optimizer_kwargs or {}
        optimizer_kwargs["lr"] = self.get("lr") or optimizer_kwargs.get("lr")
        self.optimizer = optimizer_factory(
            params=(p for model in self.models for p in model.parameters()),
            **optimizer_kwargs,
        )

    def train(self):
        # initialize
        self.problem.setup()

        # check all milestones
        mem_train, mem_test = self.check()
        # check cuda memory requirements for training and evaluation (with a small safety margin)
        if self.check_train:
            wandb.run.summary["cuda_mem_train"] = [utils.byte2mb(m) for m in mem_train]
            utils.check_mem(
                mem_train,
                mem_bound=self.max_mem_train,
                rel_margin=0.1,
                device=self.device,
            )
        if self.check_test:
            wandb.run.summary["cuda_mem_test"] = [utils.byte2mb(m) for m in mem_test]
            utils.check_mem(mem_test, rel_margin=0.1, device=self.device)

        # initial evaluation
        wandb.log(
            self.test(
                approx_generator=self.approx_gen,
                stats_generator=self.stats_gen,
                plot_generator=self.plot_gen,
            )
        )

        # training loop
        while self.step_count < self.train_steps and self.time_count < self.train_time:

            # gradient step
            metrics = self.step()

            # save models
            if self.save_freq and self.progress() >= self.save_count / self.save_freq:
                self.save()
                self.save_count += 1

            # evaluate
            if self.test_freq and self.progress() >= self.test_count / self.test_freq:
                metrics.update(
                    self.test(
                        approx_generator=self.approx_gen,
                        stats_generator=self.stats_gen,
                        plot_generator=self.plot_gen,
                    )
                )
                self.test_count += 1

            wandb.log(metrics)

    def step(self):
        start_time = time.time()
        # keep cache to not slow down training
        mem, loss = utils.max_mem_allocation(
            self._step, device=self.device, empty_cache=False
        )
        self.time_count += time.time() - start_time
        utils.check_mem(mem, mem_bound=self.max_mem_train, device=self.device)
        return {
            "train/step": self.step_count,
            "train/loss": loss.item(),
            "train/time": self.time_count,
            "train/samples": self.sample_count,
            "train/cuda_mem": utils.byte2mb(mem),
            "train/milestone": self.milestone_idx,
        }

    def _step(self):
        for model in self.models:
            model.train()

        # update lr
        for param_group in self.optimizer.param_groups:
            lr = self.get("lr") or param_group["lr"]
            param_group["lr"] = lr

        # gradient step
        loss = self.optimizer.step(
            closure=partial(self.backward, generator=self.train_gen)
        )
        self.step_count += 1
        self.sample_count += self.get("batch_size")

        # update milestones
        if (
            self.milestone_idx < len(self.milestones)
            and self.progress() >= self.milestones[self.milestone_idx]
        ):
            self.milestone_idx += 1

        return loss

    def backward(self, generator=None):
        self.optimizer.zero_grad(set_to_none=True)
        loss = simulate_loss(
            self.problem,
            self.model,
            self.get("batch_size"),
            self.get("time_step"),
            self.get("loss_mode"),
            self.get("sample_repetitions"),
            grad_model=self.grad_model,
            generator=generator,
        )
        loss.backward()
        return loss

    def test(self, approx_generator=None, plot_generator=None, stats_generator=None):
        # initialize
        metrics = {}
        for model in self.models:
            model.eval()
        plot, plot_grad = "plot" in self.visualize, "plot_grad" in self.visualize

        # approximation metrics
        if "approx" in self.visualize:
            metrics.update(
                self.evaluator.approx_metrics(density=plot, generator=approx_generator)
            )
        if plot:
            metrics.update(self.evaluator.plot(generator=plot_generator))

        # gradient approximation metrics
        if "approx_grad" in self.visualize:
            metrics.update(
                self.evaluator.approx_metrics_grad(
                    density=plot_grad, generator=approx_generator
                )
            )
        if plot_grad:
            metrics.update(self.evaluator.plot_grad(generator=plot_generator))

        # gradient and loss statistics
        if "stats" in self.visualize:
            metrics.update(
                self.evaluator.stats(
                    self.get("batch_size"),
                    self.get("time_step"),
                    self.get("loss_mode"),
                    self.get("sample_repetitions"),
                    generator=stats_generator,
                )
            )

        return metrics

    def check(self):
        # initialize
        for model in self.models:
            model.eval()
        milestone_idx = self.milestone_idx
        self.milestone_idx = 0
        mem_train, mem_test = [], []

        # check training and testing for each milestone setting and track cuda memory
        for _ in range(len(self.milestones) + 1):
            if self.check_train:
                mem, _ = utils.max_mem_allocation(self.backward, device=self.device)
                mem_train.append(mem)
            if self.check_test:
                mem, _ = utils.max_mem_allocation(self.test, device=self.device)
                mem_test.append(mem)
            self.milestone_idx += 1

        # reset milestone index
        self.milestone_idx = milestone_idx
        return mem_train, mem_test

    def progress(self):
        if self.milestone_metric == "step":
            return self.step_count / self.train_steps
        return self.time_count / self.train_time

    def get(self, attr, default=None):
        value = utils.make_iterable(getattr(self, attr, default))
        if self.milestone_idx < len(value):
            return value[self.milestone_idx]
        return value[-1]

    def save(self):
        for i, model in enumerate(self.models):
            file = os.path.join(wandb.run.dir, f"model_{i}_step_{self.step_count}.pt")
            torch.save(model.state_dict(), file)
            wandb.save(file, base_path=wandb.run.dir)


class Evaluator:
    def __init__(
        self,
        problem,
        model,
        grad_model=None,
        grad_reduce="norm",
        approx_p=1,
        approx_batch_size=131072,
        approx_batches=10,
        plot_interval_stretch=1.0,
        plot_time=None,
        plot_steps=100,
        stats_steps=30,
        stats_grad_thresh=None,
    ):

        # problem, model, and `grad_reduce_fn` mapping multi-dimensional gradients to scalars
        self.problem = problem
        self.model = model
        self.grad_model = grad_model
        self.model_gradient = self.grad_model or utils.grad_fn(
            self.model, create_graph=False
        )
        if grad_reduce == "norm":
            self.grad_reduce_fn = partial(torch.linalg.norm, dim=1, keepdim=True)
        elif grad_reduce == "projection":
            self.grad_reduce_fn = lambda grad: grad[:, 0].unsqueeze(-1)
        else:
            raise ValueError("`grad_reduce` can either be 'norm' or 'projection'.")

        # approximation metrics
        self.approx_batch_size = approx_batch_size
        self.approx_batches = approx_batches
        self.lp_metrics = utils.LpMetrics(p_values=approx_p)

        # plotting
        diam = (self.problem.interval[1] - self.problem.interval[0]) / 2
        mid = sum(self.problem.interval) / 2
        self.plot_x_axis = torch.linspace(
            mid - plot_interval_stretch * diam,
            mid + plot_interval_stretch * diam,
            steps=plot_steps,
            device="cpu",
            dtype=problem.dtype,
        )
        if plot_time is None:
            plot_time = [
                0.0,
                self.problem.terminal_time / 2,
                self.problem.terminal_time,
            ]
        self.plot_time = utils.make_iterable(plot_time)

        # statistics
        stats_grad_thresh = stats_grad_thresh or []
        self.stats_grad_thresh = {0.0}.union(utils.make_iterable(stats_grad_thresh))

        if not (isinstance(stats_steps, int) and stats_steps > 1):
            raise ValueError(
                f"`stats_steps` is {stats_steps} but needs to be an integer greater than one."
            )
        self.stats_steps = stats_steps

    @torch.no_grad()
    def approx_metrics(self, density=False, generator=None):
        self.lp_metrics.zero()
        solutions, predictions = self.accumulate_approx(
            self.problem.solution, self.model, generator=generator
        )
        metrics = self.lp_metrics.result(prefix="approx/")
        if density:
            metrics["plot/density"] = utils.plot_density(
                {"solution": solutions, "prediction": predictions}
            )
        return metrics

    def approx_metrics_grad(self, density=False, generator=None):
        self.lp_metrics.zero()
        solutions, predictions = self.accumulate_approx(
            self.problem.gradient,
            self.model_gradient,
            reduce_fn=self.grad_reduce_fn,
            generator=generator,
        )
        metrics = self.lp_metrics.result(prefix="approx_grad/")
        if density:
            metrics["plot_grad/density"] = utils.plot_density(
                {"solution": solutions, "prediction": predictions}
            )
        return metrics

    def accumulate_approx(
        self, solution_fn, prediction_fn, reduce_fn=None, generator=None
    ):
        solutions, predictions = [], []

        for _ in range(self.approx_batches):
            x, t = self.problem.sample(self.approx_batch_size, generator=generator)
            solution = solution_fn(x, t, generator=generator).detach()
            prediction = prediction_fn(x, t).detach()
            self.lp_metrics(y=solution, prediction=prediction)
            if reduce_fn:
                solution = reduce_fn(solution)
                prediction = reduce_fn(prediction)
            solutions.append(solution)
            predictions.append(prediction)

        return torch.cat(solutions), torch.cat(predictions)

    @torch.no_grad()
    def plot(self, generator=None):
        return {
            f"plot/t={t}": self.diagonal_plot(
                self.problem.solution, self.model, t, generator=generator
            )
            for t in self.plot_time
        }

    def plot_grad(self, generator=None):
        return {
            f"plot_grad/t={t}": self.diagonal_plot(
                self.problem.gradient,
                self.model_gradient,
                t,
                reduce_fn=self.grad_reduce_fn,
                generator=generator,
            )
            for t in self.plot_time
        }

    def diagonal_plot(
        self, solution_fn, prediction_fn, t, reduce_fn=None, fig=None, generator=None
    ):
        x = (
            self.plot_x_axis.to(device=self.problem.device)
            .unsqueeze(-1)
            .expand(-1, self.problem.dim)
        )
        t = torch.tensor(
            t, device=self.problem.device, dtype=self.problem.dtype
        ).expand(x.shape[0], 1)
        solution = solution_fn(x, t, generator=generator).detach()
        prediction = prediction_fn(x, t).detach()
        if reduce_fn:
            solution = reduce_fn(solution)
            prediction = reduce_fn(prediction)
        return utils.plot(
            {
                "solution": (self.plot_x_axis, solution),
                "prediction": (self.plot_x_axis, prediction),
            },
            fig=fig,
        )

    def stats(
        self,
        batch_size,
        time_step,
        loss_mode,
        repetitions,
        generator=None,
        eps=1e-06,
    ):
        losses, grads = self.accumulate_loss(
            batch_size,
            time_step,
            loss_mode,
            repetitions,
            generator=generator,
        )

        metrics = {}
        with torch.no_grad():
            # loss stats
            metrics.update(
                {
                    f"stats_loss/stddev": losses.std().item(),
                    f"stats_loss/iqr": losses.quantile(
                        torch.tensor(
                            [0.25, 0.75], device=losses.device, dtype=losses.dtype
                        )
                    )
                    .diff()
                    .item(),
                }
            )

            # gradient stats (the rel. error is also known as coefficient of variation or rel. stddev.)
            grad_stddev = grads.std(dim=0)
            grad_abs_mean = torch.abs(grads.mean(dim=0))
            grad_rel_stddev = grad_stddev / (grad_abs_mean + eps)

            for threshold in self.stats_grad_thresh:
                name = f"stats_grad_thresh{threshold}/" if threshold else "stats_grad/"
                mask = grad_abs_mean >= threshold
                grads_count = sum(mask)
                metrics[name + "numel"] = grads_count
                if not grads_count:
                    continue

                masked_grad_stddev = grad_stddev[mask]
                masked_grad_rel_stddev = grad_rel_stddev[mask]
                cos_similarity = utils.pairwise_cos_similarity(
                    grads[:, mask], grads[:, mask], eps=eps
                )

                metrics.update(
                    {
                        name + "stddev": masked_grad_stddev.detach().cpu(),
                        name + "avg_stddev": masked_grad_stddev.mean().item(),
                        name + "med_stddev": masked_grad_stddev.median().item(),
                        name + "max_stddev": masked_grad_stddev.max().item(),
                        name + "cosine_similarity": cos_similarity.detach().cpu(),
                        name + "avg_cosine_similarity": cos_similarity.mean().item(),
                        name + "med_cosine_similarity": cos_similarity.median().item(),
                        name + "min_cosine_similarity": cos_similarity.min().item(),
                        name + "rel_stddev": masked_grad_rel_stddev.detach().cpu(),
                        name + "avg_rel_stddev": masked_grad_rel_stddev.mean().item(),
                    }
                )
        return metrics

    def accumulate_loss(
        self,
        batch_size,
        time_step,
        loss_mode,
        repetitions,
        generator=None,
    ):
        grads = []
        losses = []
        for _ in range(self.stats_steps):
            # zero grads
            utils.zero_grad(self.model, set_to_none=True)
            if self.grad_model:
                utils.zero_grad(self.grad_model, set_to_none=True)

            # forward and backward pass
            loss = simulate_loss(
                self.problem,
                self.model,
                batch_size,
                time_step,
                loss_mode,
                repetitions,
                grad_model=self.grad_model,
                generator=generator,
            )
            loss.backward()

            # append losses and grads
            grad = torch.cat(
                [p.grad.view(1, -1) for p in self.model.parameters()], dim=1
            )
            losses.append(loss.detach())
            grads.append(grad.detach())
        return torch.stack(losses), torch.cat(grads)


def simulate_loss(
    problem,
    model,
    batch_size,
    time_step,
    loss_mode,
    repetitions,
    grad_model=None,
    generator=None,
):
    # gradient for simulating the stochastic integral
    detach_integral = loss_mode not in ["bsde", "model"]
    grad_fn = None
    if loss_mode in ["model", "model_eff"]:
        if not grad_model:
            raise ValueError("Gradient model not defined.")
        grad_fn = grad_model
    elif loss_mode in ["bsde", "bsde_eff", "detach"]:
        grad_fn = utils.grad_fn(model, create_graph=loss_mode == "bsde")
    elif not loss_mode == "fk":
        raise ValueError(
            "`loss_mode` can only be 'fk', 'bsde', 'bsde_eff', 'model', 'model_eff', or 'detach'."
        )

    # sample and predict
    x, t = problem.sample(batch_size, generator=generator)
    prediction = model(x, t)

    # repeat samples
    # we use `.repeat` instead of `.expand` (creating a 2nd batch dim) to allow for BatchNorm layers
    x, t = x.repeat(repetitions, 1), t.repeat(repetitions, 1)

    # temporarily deactivate training mode (except for the gradient model)
    training = model.training
    model.eval()

    # simulate SDE and stochastic integral
    state = generator.get_state() if generator else None
    sde, stoch_integral = problem.simulate(
        x,
        t,
        time_step,
        grad_fn=grad_fn,
        detach_integral=detach_integral,
        generator=generator,
    )
    y = sde - stoch_integral
    assert not sde.requires_grad

    # perform the same simulation but accumulate gradients (weighted by "cached" loss)
    if loss_mode in ["bsde_eff", "model_eff"]:
        grad_fn = (
            grad_fn
            if loss_mode == "model_eff"
            else utils.grad_fn(model, create_graph=True)
        )
        if state is not None:
            generator.set_state(state)
        problem.simulate(
            x,
            t,
            time_step,
            grad_fn,
            weights=(prediction.detach().repeat(repetitions, 1) - y),
            detach_integral=True,
            generator=generator,
        )

    # reactivate the previous mode
    model.train(training)

    # average repetitions and compute point-wise error
    assert y.requires_grad is not detach_integral
    return torch.nn.MSELoss()(prediction, y.view(-1, batch_size, 1).mean(dim=0))


def solve(config=None, overwrite_config=None):
    with wandb.init(config=config):
        if overwrite_config is not None:
            wandb.config.update(overwrite_config, allow_val_change=True)
        config = dict(wandb.config)

        # import classes
        if "import_keys" in config:
            for path in config.pop("import_keys"):
                if path in config:
                    config[path] = utils.import_string(config[path])

        # solve the problem
        # we nest the config now as this is currently not yet supported by wandb sweeps
        # see https://github.com/wandb/client/issues/2005
        solver = Solver(**utils.nest_dict(config))
        err, _ = utils.catch_oom(solver.train)
        wandb.run.summary["cuda_oom"] = bool(err)
        if err:
            raise err
