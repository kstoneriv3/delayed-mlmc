import logging
import math
import os
import time
from contextlib import contextmanager
from datetime import datetime
from functools import partial, wraps
from pathlib import Path
from typing import Any, Callable, List, Optional, ParamSpec, Tuple, Union, cast

# Showing the compilation logs
os.environ["JAX_LOG_COMPILES"] = "1"  # noqa
# The below does not work as it just emulates devices for debugging purpose.
# It seems that only way to use multiprocessing with jax is manual MPI by mpi4jax.
# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=7"

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import jax_smi
import matplotlib.pyplot as plt
import mlflow
import mpi4jax
import numpy as np
import optax
import polars as pl
from diffrax import (
    AbstractBrownianPath,
    ControlTerm,
    Euler,
    ItoMilstein,
    MultiTerm,
    ODETerm,
    SaveAt,
    StepTo,
    diffeqsolve,
)
from diffrax.custom_types import Int, Scalar
from jaxtyping import Array, Float, PRNGKeyArray, PyTree
from mpi4py import MPI
from tqdm import tqdm, trange

# With CPU, it takes an hour to compile, and 5 mins per iteration
# jax.config.update("jax_platform_name", "cpu")  # type: ignore[no-untyped-call]


T0, T1, N_STEPS_LEVEL0, DIM = 0, 1, 10, 1

BATCH_SIZE, LR, N_ITER, MAX_LEVEL = 2**11, 1e-3, 1000, 7
# BATCH_SIZE, LR, N_ITER, MAX_LEVEL = 2**9, 1e-3, 400, 6
# BATCH_SIZE, LR, N_ITER, MAX_LEVEL = 2**5, 1e-2, 20, 3

KEY = jr.PRNGKey(3)
MU = 1.0
SIGMA = 1.0
STRIKE_PRICE = jnp.exp(MU)

# N_REPEAT_EXPERIMENT = 10
N_REPEAT_EXPERIMENT = 1
COST_RATE = 1
VARIANCE_DECAY_RATE = 1.2
# VARIANCE_DECAY_RATE = 2
SMOOTHNESS_DECAY_RATE = 1.0
# SMOOTHNESS_DECAY_RATE = 1

WORLD_RANK = MPI.COMM_WORLD.Get_rank()
WORLD_SIZE = MPI.COMM_WORLD.Get_size()

assert WORLD_SIZE == 1 or WORLD_SIZE == MAX_LEVEL + 1


@contextmanager  # type: ignore[arg-type]
def nullcontext(*args: Any, **kwargs: Any) -> None:  # type: ignore
    yield


def nullfunc(*args: Any, **kwargs: Any) -> None:
    pass


if WORLD_RANK != 0:
    mlflow.start_run = nullcontext
    mlflow.log_params = nullfunc
    mlflow.log_metric = nullfunc
    mlflow.set_experiment = nullfunc


def mpi_breakpoint(rank: int = 0) -> None:
    if WORLD_RANK == rank:
        breakpoint()  # noqa


def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d%H%M%S")


def save_array(array: Array, path: Path) -> None:
    if WORLD_RANK != 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def tree_zeros_like(tree: PyTree) -> PyTree:
    return jtu.tree_map(jnp.zeros_like, tree)


def sum_if_arrays(*args: Array) -> Array:
    assert all(isinstance(a, Array) for a in args)
    assert all(arr.shape == args[0].shape for arr in args[1:])
    return cast(Array, sum(args))


def tree_sum(trees: List[PyTree]) -> PyTree:
    return jtu.tree_map(sum_if_arrays, *trees)


def tree_mpi_reduce_sum(trees: List[PyTree]) -> PyTree:
    assert len(trees) == WORLD_SIZE
    tree = trees[WORLD_RANK]
    mpi_reduce_sum = lambda x: mpi4jax.allreduce(x, op=MPI.SUM, comm=MPI.COMM_WORLD)[0]  # noqa
    return jtu.tree_map(mpi_reduce_sum, tree)


def clear_all_jit_cache() -> None:
    for k, v in globals().items():
        try:
            v._cached._clear_cache()
        except:
            pass


class FixedBrownianMotion(AbstractBrownianPath):
    bm: Float[Array, "n_steps+1 dim"]  #  = eqx.field(static=True) # to avoid leaked tracer error
    t0: float = eqx.field(static=True)
    t1: float = eqx.field(static=True)
    n_steps: int = eqx.field(static=True)

    def evaluate(
        self, t0: Scalar, t1: Optional[Scalar] = None, left: bool = True
    ) -> Float[Array, "dim"]:
        del left
        if t1 is not None:
            return self.evaluate(t1) - self.evaluate(t0)
        idx = (self.n_steps * (t0 - self.t0)).astype(int)
        return self.bm[idx]

    @classmethod
    def create_from_interval(
        cls,
        t0: Scalar,
        t1: Scalar,
        n_steps: Int,
        dim: Int,
        key: PRNGKeyArray,
    ) -> "FixedBrownianMotion":
        rnorm = jr.normal(key, (n_steps, dim))
        bm = jnp.cumsum(
            ((t1 - t0) / n_steps) ** 0.5 * jnp.concatenate([jnp.zeros((1, dim)), rnorm], axis=0),
            axis=0,
        )
        return cls(bm=bm, t0=t0, t1=t1, n_steps=n_steps)

    def to_antithetic_path(self) -> "FixedBrownianMotion":
        assert self.n_steps % 2 == 0
        bm = self.bm.at[1::2].set(self.bm[::2][:-1] + self.bm[::2][1:] - self.bm[1:][::2])
        return FixedBrownianMotion(bm=bm, t0=self.t0, t1=self.t1, n_steps=self.n_steps)


class HoldingStrategy(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, dim: Int, key: PRNGKeyArray) -> None:
        self.mlp = eqx.nn.MLP(
            in_size=1 + dim,
            out_size=dim,
            width_size=32,
            depth=4,
            activation=jax.nn.silu,
            final_activation=jax.nn.sigmoid,
            key=key,
        )

    @jax.named_scope("HoldingStrategy")
    def __call__(self, t: Scalar, y: Float[Array, "dim"]) -> Float[Array, "dim"]:
        return self.mlp(jnp.concatenate([t[None], y]))


def BlackScholesDrift(t: Scalar, y: Float[Array, "dim"]) -> Float[Array, "dim"]:
    return MU * y


def BlackScholesDiffusion(t: Scalar, y: Float[Array, "dim"]) -> Float[Array, "dim dim"]:
    return jnp.diag(SIGMA * y)  # type: ignore[no-untyped-call]


class DeepHedgingLoss(eqx.Module):
    """Implements equation 4.6 of [1], whose expectation is minimized for deep hedging.

    [1] Buehler, H., Gonon, L., Teichmann, J., & Wood, B. (2019). Deep hedging.
    Quantitative Finance, 19(8), 1271-1291.
    """

    w: Float[Array, ""]  # indifference price
    h: eqx.Module  # holding function delta = h(t, s) that takes values in [0, 1].
    s_drift: Callable = eqx.field(static=True)
    s_diffusion: Callable = eqx.field(static=True)
    dim: int = eqx.field(static=True)

    @classmethod
    def create_from_dim_and_key(cls, dim: Int, key: PRNGKeyArray) -> "DeepHedgingLoss":
        assert dim == 1
        h = HoldingStrategy(dim, key)
        w = jnp.array(0.0)
        return cls(h=h, w=w, s_drift=BlackScholesDrift, s_diffusion=BlackScholesDiffusion, dim=dim)

    def drift(self, t: Scalar, y: Float[Array, "dim+1"], args: Any) -> Float[Array, "dim+1"]:
        # We can only handle cases where dim = 1.
        s = y[: self.dim]
        dhdt = jax.jacobian(self.h, argnums=0)(t, s)
        dhds = jax.jacobian(self.h, argnums=1)(t, s)[:, 0]
        dhdss = jax.hessian(self.h, argnums=1)(t, s)[:, 0, 0]
        s_drift = self.s_drift(t, s)
        s_diffusion = jnp.diag(self.s_diffusion(t, s))  # type: ignore[no-untyped-call]
        h_drift = dhdt + dhds * s_drift + dhdss * s_diffusion**2 / 2
        return jnp.concatenate([s_drift, s * h_drift], axis=0)

    def diffusion(
        self, t: Scalar, y: Float[Array, "dim+1"], args: Any
    ) -> Float[Array, "dim+1 dim"]:
        # We can only handle cases where dim = 1.
        s = y[: self.dim]
        dhds = jax.jacobian(self.h, argnums=1)(t, s)[:, 0]
        s_diffusion = self.s_diffusion(t, s)
        h_diffusion = dhds * s_diffusion
        return jnp.concatenate([s_diffusion, s * h_diffusion], axis=0)

    @jax.named_scope("DeepHedgingLoss")
    def __call__(
        self, bm: AbstractBrownianPath, ts: Float[Array, " n_steps+1"]
    ) -> Float[Array, ""]:
        terms = MultiTerm(ODETerm(self.drift), ControlTerm(self.diffusion, bm))
        solver = ItoMilstein()
        # solver = Euler()
        saveat = SaveAt(t1=True)
        stepto = StepTo(ts)
        y0 = jnp.array([1.0, 0.0])

        sol = diffeqsolve(
            terms,
            solver,
            T0,
            T1,
            dt0=None,
            y0=y0,
            saveat=saveat,
            stepsize_controller=stepto,
        )
        s1 = sol.ys[0, 0]
        z = jnp.maximum(0, s1 - STRIKE_PRICE)
        hedging_pnl = sol.ys[0, 1]
        loss = lambda z: (z**2 + 1) / 2  # noqa
        w = 10 * self.w  # rescale to match learning rate with NN
        J = w + loss(z - hedging_pnl - w)  # type: ignore[no-untyped-call]
        return J


def save_model_weight(model: DeepHedgingLoss, path: Path) -> None:
    if WORLD_RANK != 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(path, model)
    mlflow.log_artifacts(path.parent.absolute().as_posix(), path.name)


def load_model_weight(model: DeepHedgingLoss, path: Path) -> DeepHedgingLoss:
    return eqx.tree_deserialise_leaves(path, model)


@eqx.filter_jit
@partial(jax.vmap, in_axes=(None, 0, None))
def batched_loss_baseline(
    model: DeepHedgingLoss, key: PRNGKeyArray, max_level: int
) -> Float[Array, ""]:
    # bm = VirtualBrownianTree(T0, T1, tol=0.001, shape=(DIM,), key=key)
    n_steps = N_STEPS_LEVEL0 * 2**max_level
    bm = FixedBrownianMotion.create_from_interval(T0, T1, n_steps, DIM, key)
    ts = (T1 - T0) * jnp.arange(n_steps + 1) / n_steps + T0
    return model(bm, ts)


@eqx.filter_jit
@eqx.filter_value_and_grad
def loss_and_grad_baseline(
    model: DeepHedgingLoss, keys: PRNGKeyArray, max_level: int
) -> Float[Array, ""]:
    return jnp.mean(batched_loss_baseline(model, keys, max_level))


@eqx.filter_jit
@partial(jax.vmap, in_axes=(None, 0, None))
def batched_loss(model: DeepHedgingLoss, key: PRNGKeyArray, level: int) -> Float[Array, ""]:
    n_steps = N_STEPS_LEVEL0 * 2**level
    bm = FixedBrownianMotion.create_from_interval(T0, T1, n_steps, DIM, key)
    # bm = VirtualBrownianTree(T0, T1, tol=0.001, shape=(DIM,), key=key)
    # abm = bm.to_antithetic_path()
    ts = (T1 - T0) * jnp.arange(n_steps + 1) / n_steps + T0
    ts_coarse = ts[::2]
    if level == 0:
        return model(bm, ts)
    else:
        # return (model(bm, ts) + model(abm, ts)) / 2 - model(bm, ts_coarse)
        return model(bm, ts) - model(bm, ts_coarse)


@eqx.filter_jit
@eqx.filter_value_and_grad
def loss_and_grad(model: DeepHedgingLoss, keys: PRNGKeyArray, level: int) -> Float[Array, ""]:
    print()
    print(f"loss_and_grad traced for level {level}")
    print(jax._src.lax.control_flow._initial_style_jaxprs_with_common_consts.cache_info())  # type: ignore
    print()
    return jnp.mean(batched_loss(model, keys, level))


@eqx.filter_jit
@partial(jax.vmap, in_axes=(None, 0, None))
def grad_l2_norm(model: DeepHedgingLoss, keys: PRNGKeyArray, level: int) -> Float[Array, ""]:
    l, g = loss_and_grad(model, keys, level)  # noqa
    squared_sum = sum(jnp.sum(grad**2) for grad in jtu.tree_leaves(g))
    return squared_sum**0.5


@eqx.filter_jit
@partial(jax.vmap, in_axes=(None, None, 0, None))
def grad_diff_l2_norm(
    model0: DeepHedgingLoss, model1: DeepHedgingLoss, keys: PRNGKeyArray, level: int
) -> Float[Array, ""]:
    l0, g0 = loss_and_grad(model0, keys, level)  # noqa
    l1, g1 = loss_and_grad(model1, keys, level)  # noqa
    g_pairs = zip(jtu.tree_leaves(g0), jtu.tree_leaves(g1))
    squared_sum = sum(jnp.sum((g_pair[0] - g_pair[1]) ** 2) for g_pair in g_pairs)
    return squared_sum**0.5


@eqx.filter_jit
def param_diff_l2_norm(model0: DeepHedgingLoss, model1: DeepHedgingLoss) -> Float[Array, ""]:
    param0 = jtu.tree_leaves(eqx.filter(model0, eqx.is_inexact_array))
    param1 = jtu.tree_leaves(eqx.filter(model1, eqx.is_inexact_array))
    squared_sum = sum(jnp.sum((p1 - p2) ** 2) for p1, p2 in zip(param0, param1))
    return squared_sum**0.5


def log_grad_l2_norms(model: DeepHedgingLoss, save_path: Path, key: PRNGKeyArray) -> None:
    norms_outer = []
    for level in trange(MAX_LEVEL, -1, -1, desc=f"Evaluating variance"):
        norms_inner = []
        for i in trange(2**8, desc=f"Level {level}", leave=False):
            keys_2d = jr.split(jr.fold_in(key, i), (2**8, 1))
            norms_inner.append(grad_l2_norm(model, keys_2d, level))
        norms_outer.append(jnp.concatenate(norms_inner))
        clear_all_jit_cache()
        jax.lib.xla_bridge.get_backend().defragment()
    norms = jnp.stack(norms_outer[::-1])
    fp = save_path / "grad_l2_norms.npy"
    save_array(norms, fp)
    mlflow.log_artifacts(fp.parent.absolute().as_posix(), fp.name)


@eqx.filter_jit
def perturb_model(model: DeepHedgingLoss, key: PRNGKeyArray) -> DeepHedgingLoss:
    keys = jr.split(key, 2**8)
    loss, grad = loss_and_grad(model, keys, 0)
    update = jtu.tree_map(lambda x: -1e-4 * x, grad)
    return eqx.apply_updates(model, update)


def get_perturbed_models(
    model: DeepHedgingLoss, key: PRNGKeyArray, n_models: int
) -> List[DeepHedgingLoss]:
    models = []
    for i in trange(n_models, desc="Perturbing models", leave=False):
        models.append(perturb_model(model, jr.fold_in(key, i)))
    clear_all_jit_cache()
    jax.lib.xla_bridge.get_backend().defragment()
    return models


def log_normalized_grad_diff_l2_norms(
    model: DeepHedgingLoss,
    save_path: Path,
    key: PRNGKeyArray,
) -> None:
    models = get_perturbed_models(model, key, 2**8)
    norms_outer = []
    for level in trange(MAX_LEVEL, -1, -1, desc=f"Evaluating smoothness"):
        norms_inner = []
        for i, m in enumerate(tqdm(models, desc=f"Level {level}", leave=False)):
            keys_2d = jr.split(jr.fold_in(key, i), (2**7, 1))
            norms_inner.append(
                grad_diff_l2_norm(model, m, keys_2d, level) / param_diff_l2_norm(model, m)
            )
        norms_outer.append(jnp.stack(norms_inner))
        clear_all_jit_cache()
        jax.lib.xla_bridge.get_backend().defragment()
    norms = jnp.stack(norms_outer[::-1])
    fp = save_path / "normalized_grad_diff_l2_norms.npy"
    save_array(norms, fp)
    mlflow.log_artifacts(fp.parent.absolute().as_posix(), fp.name)


@eqx.filter_jit
def step_baseline(
    model: DeepHedgingLoss,
    optim: optax.GradientTransformation,
    opt_state: optax.OptState,
    key: PRNGKeyArray,
    max_level: int,
) -> Tuple[Float[Array, ""], DeepHedgingLoss, optax.OptState]:
    keys = jr.split(key, BATCH_SIZE)
    loss, grad = loss_and_grad_baseline(model, keys, max_level)
    updates, opt_state = optim.update(grad, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


@eqx.filter_jit
def _step_from_grad_per_level(
    model: DeepHedgingLoss,
    optim: optax.GradientTransformation,
    opt_state: optax.OptState,
    grad_per_level: List[DeepHedgingLoss],
) -> Tuple[DeepHedgingLoss, optax.OptState]:
    if WORLD_SIZE == 1:
        grad_sum = tree_sum(grad_per_level)
    else:
        grad_sum = tree_mpi_reduce_sum(grad_per_level)
    updates, opt_state = optim.update(grad_sum, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state


def step_mlmc(
    model: DeepHedgingLoss,
    optim: optax.GradientTransformation,
    opt_state: optax.OptState,
    key: PRNGKeyArray,
) -> Tuple[Float[Array, ""], DeepHedgingLoss, optax.OptState]:
    grad_per_level = [None] * (MAX_LEVEL + 1)
    for level in range(MAX_LEVEL, -1, -1):
        if WORLD_SIZE != 1 and level != WORLD_RANK:
            continue
        batch_size = math.ceil(BATCH_SIZE // 2 ** ((VARIANCE_DECAY_RATE - COST_RATE) * level))
        keys = jr.split(key, batch_size)
        loss, grad = loss_and_grad(model, keys, level)
        grad_per_level[level] = grad
    model, opt_state = _step_from_grad_per_level(model, optim, opt_state, grad_per_level)
    if WORLD_SIZE != 1 and WORLD_RANK != 0:
        loss = 0.0
    else:
        loss = jnp.mean(batched_loss_baseline(model, jr.split(key, BATCH_SIZE), MAX_LEVEL))
    return loss, model, opt_state


def step_delayed_mlmc(
    model: DeepHedgingLoss,
    optim: optax.GradientTransformation,
    opt_state: optax.OptState,
    grad_per_level: List[DeepHedgingLoss],
    key: PRNGKeyArray,
    step: int,
) -> Tuple[Float[Array, ""], DeepHedgingLoss, optax.OptState, List[DeepHedgingLoss]]:
    keys = jr.split(key, BATCH_SIZE)
    # periods = [math.ceil(2 ** (SMOOTHNESS_DECAY_RATE * l)) for l in range(MAX_LEVEL + 1)]
    periods = [
        math.floor(2 ** (1 + SMOOTHNESS_DECAY_RATE * (level - 1))) for level in range(MAX_LEVEL + 1)
    ]
    for level in range(MAX_LEVEL, -1, -1):
        if step % periods[level] != 0:
            continue
        if WORLD_SIZE != 1 and level != WORLD_RANK:
            continue
        batch_size = math.ceil(BATCH_SIZE // 2 ** ((VARIANCE_DECAY_RATE - COST_RATE) * level))
        keys = jr.split(key, batch_size)
        loss, grad = loss_and_grad(model, keys, level)
        grad_per_level[level] = grad

    model, opt_state = _step_from_grad_per_level(model, optim, opt_state, grad_per_level)
    if WORLD_SIZE != 1 and WORLD_RANK != 0:
        loss = 0.0
    else:
        loss = jnp.mean(batched_loss_baseline(model, jr.split(key, BATCH_SIZE), MAX_LEVEL))
    return loss, model, opt_state, grad_per_level


def run_deep_hedging() -> None:
    """Run deep hedging N_REPEAT_EXPERIMENT times for baseline, mlmc, and delayed-mlmc method."""

    mlflow.set_experiment("run_deep_hedging")
    timestamp = get_timestamp()
    save_path = Path("./logs") / timestamp

    losses_all = []
    times_all = []
    # for method in tqdm(["baseline"]):
    # for method in tqdm(["mlmc"]):
    # for method in tqdm(["delayed_mlmc"]):
    # for method in tqdm(["delayed_mlmc", "baseline"]):
    for method in tqdm(["delayed_mlmc", "mlmc", "baseline"]):
        losses_outer = []
        times_outer = []
        for n in trange(N_REPEAT_EXPERIMENT, desc=f"Using {method}", leave=False):
            key_outer, model_key = jr.split(jr.fold_in(KEY, n), 2)
            model = model_prev = DeepHedgingLoss.create_from_dim_and_key(DIM, model_key)
            models = [model, model_prev]

            optim = optax.sgd(LR)
            opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
            if method == "delayed_mlmc":
                grad_per_level = [
                    tree_zeros_like(eqx.filter(model, eqx.is_array)) for _ in range(MAX_LEVEL + 1)
                ]

            losses_inner = []
            times_inner = []
            with mlflow.start_run(run_name=f"{method}-{timestamp}"):
                global_params = {
                    k: v
                    for k, v in globals().items()
                    if k.upper() == k and isinstance(v, (int, float, str))
                }
                global_params.update({"KEY": int(KEY[1])})  # type: ignore
                mlflow.log_params(global_params)
                mlflow.log_params({"method": method, "key_outer": key_outer})
                n_iter = 5 * N_ITER if method == "delayed_mlmc" else N_ITER
                pbar = trange(n_iter, leave=False)
                for i in pbar:
                    key_inner = jr.fold_in(key_outer, i)
                    model_prev = model
                    match method:
                        case "baseline":
                            loss, model, opt_state = step_baseline(
                                model, optim, opt_state, key_inner, MAX_LEVEL
                            )
                        case "mlmc":
                            loss, model, opt_state = step_mlmc(model, optim, opt_state, key_inner)
                        case "delayed_mlmc":
                            loss, model, opt_state, grad_per_level = step_delayed_mlmc(
                                model, optim, opt_state, grad_per_level, key_inner, i
                            )
                    losses_inner.append(loss)
                    times_inner.append(time.time())
                    pbar.set_description(desc="Step: {:>3d}, Loss: {:>5.2f}".format(i, float(loss)))
                    mlflow.log_metric("loss", float(loss), i)
                losses_outer.append(jnp.stack(losses_inner))
                times_outer.append(jnp.stack(losses_inner))
                save_model_weight(model, save_path / f"model_{method}_{i}.eqx")

        save_array(jnp.stack(losses_outer), save_path / f"losses_{method}.npy")
        save_array(jnp.stack(times_outer), save_path / f"times_{method}.npy")
        losses_all.append(jnp.stack(losses_outer))
        times_all.append(jnp.stack(times_outer))

    save_array(jnp.stack(losses_all), save_path / f"losses_all.npy")
    save_array(jnp.stack(times_all), save_path / f"times_all.npy")


def examine_mlmc_decay() -> None:
    """Record the variance and smoothness per level during the optimization."""
    mlflow.set_experiment("examine_mlmc_decay")

    # timestamp = get_timestamp()
    timestamp = "20230928004907"

    save_path = Path("./logs") / timestamp
    key, model_key = jr.split(KEY, 2)
    model = model_prev = DeepHedgingLoss.create_from_dim_and_key(DIM, model_key)

    load_model_weight(model, save_path / "model.eqx")
    log_normalized_grad_diff_l2_norms(model, save_path / "after", key)
    log_grad_l2_norms(model, save_path / "after", key)
    exit()

    # log_normalized_grad_diff_l2_norms(model, save_path / "before", key)
    # log_grad_l2_norms(model, save_path / "before", key)

    with mlflow.start_run():
        global_params = {
            k: v
            for k, v in globals().items()
            if k.upper() == k and isinstance(v, (int, float, str))
        }
        global_params.update({"KEY": int(KEY[1])})  # type: ignore
        mlflow.log_params(global_params)

        optim = optax.sgd(LR)
        opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
        losses = []
        pbar = trange(N_ITER + 1)
        for i in pbar:
            key = jr.fold_in(key, i)
            loss, model, opt_state = step_baseline(model, optim, opt_state, key, 0)
            losses.append(loss)
            pbar.set_description(desc="Step: {:>3d}, Loss: {:>5.2f}".format(i, loss))
            mlflow.log_metric("loss", float(loss), i)

        save_array(jnp.stack(losses), save_path / "losses.npy")
        save_model_weight(model, save_path / "model.eqx")

        log_grad_l2_norms(model, save_path / "after", key)
        log_normalized_grad_diff_l2_norms(model, save_path / "after", key)


if __name__ == "__main__":
    logging.getLogger("jax").setLevel(logging.INFO)
    jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]
    jax_smi.initialise_tracking()
    # jax.experimental.compilation_cache.compilation_cache.initialize_cache(
    #     "./.compilation_cache"
    # )  # type: ignore[no-untyped-call]
    # jax.experimental.io_callback()
    with jax.disable_jit(False):
        examine_mlmc_decay()
        # run_deep_hedging()
