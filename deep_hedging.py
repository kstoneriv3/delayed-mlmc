import logging
import math
from datetime import datetime
from functools import partial, wraps
from pathlib import Path
from typing import Any, Callable, List, Optional, ParamSpec, Tuple, TypeVar, Union, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax_smi
import matplotlib.pyplot as plt
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
from tqdm import tqdm, trange

# With CPU, it takes an hour to compile, and 5 mins per iteration
jax.config.update("jax_platform_name", "cpu")  # type: ignore[no-untyped-call]

T0, T1, N_STEPS, DIM = 0, 1, 10, 1

# BATCH_SIZE, LR, N_ITER, MAX_LEVEL = 2**11, 1e-2, 1000, 7
BATCH_SIZE, LR, N_ITER, MAX_LEVEL = 2**11, 1e-2, 10, 1

KEY = jr.PRNGKey(0)
MU = 1.0
SIGMA = 1.0
STRIKE_PRICE = jnp.exp(MU)

# N_REPEAT_EXPERIMENT = 10
N_REPEAT_EXPERIMENT = 1
COST_RATE = 1
# VARIANCE_DECAY_RATE = 1.5
VARIANCE_DECAY_RATE = 2
# SMOOTHNESS_DECAY_RATE = 0.7
SMOOTHNESS_DECAY_RATE = 1


def simple_filter_jit(fun):  # type: ignore
    @partial(jax.jit, static_argnums=1)
    def fun_jitted(dynamic, static):  # type: ignore
        args, kwargs = eqx._compile_utils.hashable_combine(dynamic, static)
        out = fun(*args, **kwargs)
        dynamic_out, static_out = eqx._filters.partition(out, eqx.is_array)
        return dynamic_out, eqx._module.Static(static_out)

    @wraps(fun)
    def fun_new(*args, **kwargs):  # type: ignore
        dynamic, static = eqx._compile_utils.hashable_partition((args, kwargs), eqx.is_array)
        dynamic_out, static_out = fun_jitted(dynamic, static)
        return eqx._filters.combine(dynamic_out, static_out.value)

    return fun_new


def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d%H%M%S")


def save_array(array: Array, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def tree_zeros_like(tree: PyTree) -> PyTree:
    return jax.tree_util.tree_map(jnp.zeros_like, tree, eqx.is_array)


def array_sum(*arrays: Array) -> Array:
    assert all(arr is not None for arr in arrays)
    assert all(arr.shape == arrays[0].shape for arr in arrays[1:])
    return cast(Array, sum(arrays))


def tree_sum(*trees: PyTree) -> PyTree:
    return jax.tree_util.tree_map(array_sum, *trees, eqx.is_array)


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


@eqx.filter_jit
@partial(jax.vmap, in_axes=(None, 0, None))
def batched_loss_baseline(model: DeepHedgingLoss, key: PRNGKeyArray) -> Float[Array, ""]:
    # bm = VirtualBrownianTree(T0, T1, tol=0.001, shape=(DIM,), key=key)
    bm = FixedBrownianMotion.create_from_interval(T0, T1, N_STEPS * 2**MAX_LEVEL, DIM, key)
    ts = (T1 - T0) * jnp.arange(0, N_STEPS * 2**MAX_LEVEL + 1) / (N_STEPS * 2**MAX_LEVEL) + T0
    return model(bm, ts)


@eqx.filter_jit
@eqx.filter_value_and_grad
def loss_and_grad_baseline(model: DeepHedgingLoss, keys: PRNGKeyArray) -> Float[Array, ""]:
    return jnp.mean(batched_loss_baseline(model, keys))


@eqx.filter_jit
@partial(jax.vmap, in_axes=(None, 0, None))
def batched_loss(model: DeepHedgingLoss, key: PRNGKeyArray, level: int) -> Float[Array, ""]:
    # bm = VirtualBrownianTree(T0, T1, tol=0.001, shape=(DIM,), key=key)
    bm = FixedBrownianMotion.create_from_interval(T0, T1, N_STEPS * 2**level, DIM, key)
    # abm = bm.to_antithetic_path()
    ts = (T1 - T0) * jnp.arange(0, N_STEPS * 2**level + 1) / (N_STEPS * 2**level) + T0
    ts_coarse = ts[::2]
    if level == 0:
        return model(bm, ts)
    else:
        # return (model(bm, ts) + model(abm, ts)) / 2 - model(bm, ts_coarse)
        return model(bm, ts) - model(bm, ts_coarse)


@eqx.filter_jit
@eqx.filter_value_and_grad
def loss_and_grad(model: DeepHedgingLoss, keys: PRNGKeyArray, level: int) -> Float[Array, ""]:
    return jnp.mean(batched_loss(model, keys, level))


# @simple_filter_jit
@eqx.filter_jit
@partial(jax.vmap, in_axes=(None, 0, None))
def grad_l2_norm(model: DeepHedgingLoss, keys: PRNGKeyArray, level: int) -> Float[Array, ""]:
    l, g = loss_and_grad(model, keys, level)  # noqa
    squared_sum = sum(jnp.sum(grad**2) for grad in jax.tree_util.tree_leaves(g))
    return squared_sum**0.5


# @simple_filter_jit
@eqx.filter_jit
@partial(jax.vmap, in_axes=(None, None, 0, None))
def grad_diff_l2_norm(
    model0: DeepHedgingLoss, model1: DeepHedgingLoss, keys: PRNGKeyArray, level: int
) -> Float[Array, ""]:
    l0, g0 = loss_and_grad(model0, keys, level)  # noqa
    l1, g1 = loss_and_grad(model1, keys, level)  # noqa
    g_pairs = zip(jax.tree_util.tree_leaves(g0), jax.tree_util.tree_leaves(g1))
    squared_sum = sum(jnp.sum((g_pair[0] - g_pair[1]) ** 2) for g_pair in g_pairs)
    return squared_sum**0.5


@eqx.filter_jit
def param_diff_l2_norm(model0: DeepHedgingLoss, model1: DeepHedgingLoss) -> Float[Array, ""]:
    param0 = jax.tree_util.tree_leaves(eqx.filter(model0, eqx.is_inexact_array))
    param1 = jax.tree_util.tree_leaves(eqx.filter(model1, eqx.is_inexact_array))
    squared_sum = sum(jnp.sum((p1 - p2) ** 2) for p1, p2 in zip(param0, param1))
    return squared_sum**0.5


def log_grad_l2_norms(model: DeepHedgingLoss, save_path: Path, key: PRNGKeyArray) -> None:
    norms_outer = []
    for l in trange(MAX_LEVEL + 1, 0, -1, desc=f"Evaluating variance"):
        norms_inner = []
        for i in trange(2**5, desc=f"Level {l}", leave=False):
            keys_2d = jr.split(jr.fold_in(key, 2**5 * l + i), (2**8, 1))
            norms_inner.append(grad_l2_norm(model, keys_2d, l))
        norms_outer.append(jnp.concatenate(norms_inner))
        grad_l2_norm._cached._clear_cache()
    norms = jnp.stack(norms_outer)
    save_array(norms, save_path / "grad_l2_norms.npy")


@eqx.filter_jit
def perturb_model(model: DeepHedgingLoss, key: PRNGKeyArray) -> DeepHedgingLoss:
    keys = jr.split(key, 2**8)
    loss, grad = loss_and_grad(model, keys, 0)
    update = jax.tree_util.tree_map(lambda x: -1e-4 * x, grad)
    return eqx.apply_updates(model, update)


def get_perturbed_models(
    model: DeepHedgingLoss, key: PRNGKeyArray, n_models: int
) -> List[DeepHedgingLoss]:
    models = []
    for i in trange(2**5, desc="Perturbing models", leave=False):
        models.append(perturb_model(model, jr.fold_in(key, i)))
    perturb_model._cached._clear_cache()
    return models


def log_normalized_grad_diff_l2_norms(
    model: DeepHedgingLoss,
    save_path: Path,
    key: PRNGKeyArray,
) -> None:
    models = get_perturbed_models(model, key, 2**5)
    norms_outer = []
    for l in trange(MAX_LEVEL + 1, 0, -1, desc=f"Evaluating smoothness"):
        norms_inner = []
        for i, m in enumerate(tqdm(models, desc=f"Level {l}", leave=False)):
            keys_2d = jr.split(jr.fold_in(key, 2**5 * l + i), (2**7, 1))
            norms_inner.append(
                grad_diff_l2_norm(model, m, keys_2d, l) / param_diff_l2_norm(model, m)
            )
        norms_outer.append(jnp.stack(norms_inner))
        grad_diff_l2_norm._cached._clear_cache()
    norms = jnp.stack(norms_outer)
    save_array(norms, save_path / "normalized_grad_diff_l2_norms.npy")


@eqx.filter_jit
def step_baseline(
    model: DeepHedgingLoss,
    opt_state: optax.OptState,
    optim: optax.GradientTranformation,
    key: PRNGKeyArray,
) -> Tuple[DeepHedgingLoss, optax.OptState]:
    keys = jr.split(key, BATCH_SIZE)
    loss, grad = loss_and_grad_baseline(model, keys, MAX_LEVEL)
    updates, opt_state = optim.update(grad, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state


@eqx.filter_jit
def _step_from_grad_per_level(
    model: DeepHedgingLoss,
    opt_state: optax.OptState,
    optim: optax.GradientTranformation,
    grad_per_level: List[DeepHedgingLoss],
) -> Tuple[DeepHedgingLoss, optax.OptState]:
    grad_sum = tree_sum(grad_per_level)
    updates, opt_state = optim.update(grad_sum, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state


def step_mlmc(
    model: DeepHedgingLoss,
    opt_state: optax.OptState,
    optim: optax.GradientTranformation,
    key: PRNGKeyArray,
) -> Tuple[DeepHedgingLoss, optax.OptState]:
    grad_per_level = []
    for level in range(MAX_LEVEL + 1):
        keys = jr.split(key, BATCH_SIZE // 2 ** ((VARIANCE_DECAY_RATE - COST_RATE) * level))
        loss, grad = loss_and_grad(model, keys, level)
        grad_per_level.append(grad)
    return _step_from_grad_per_level(model, opt_state, optim, grad_per_level)


def step_delayed_mlmc(
    model: DeepHedgingLoss,
    optim: optax.GradientTranformation,
    opt_state: optax.OptState,
    grad_per_level: List[DeepHedgingLoss],
    key: PRNGKeyArray,
    step: int,
) -> Tuple[DeepHedgingLoss, optax.OptState, List[DeepHedgingLoss]]:
    keys = jr.split(key, BATCH_SIZE)
    for level in range(MAX_LEVEL + 1):
        # if step % math.ceil(2 ** (SMOOTHNESS_DECAY_RATE * level)) == 0:
        if step % math.floor(2 ** (1 + SMOOTHNESS_DECAY_RATE * (level - 1))) == 0:
            keys = jr.split(key, BATCH_SIZE // 2 ** ((VARIANCE_DECAY_RATE - COST_RATE) * level))
            loss, grad = loss_and_grad(model, keys, level)
            grad_per_level[level] = grad
    model, opt_state = _step_from_grad_per_level(model, opt_state, optim, grad_per_level)
    return model, opt_state, grad_per_level


def run_deep_hedging() -> None:
    """Run deep hedging N_REPEAT_EXPERIMENT times for baseline, mlmc, and delayed-mlmc method."""

    timestamp = get_timestamp()
    save_path = Path("./logs") / timestamp

    losses_all = []
    for step_func in tqdm([step_mlmc, step_delayed_mlmc, step_baseline]):
        method = step_func.__name__.removeprefix("step_")
        losses_outer = []
        for n in trange(N_REPEAT_EXPERIMENT, desc=f"Using {method}", leave=False):
            key_outer, model_key = jr.split(jr.fold_in(KEY, n), 2)
            model = model_prev = DeepHedgingLoss.create_from_dim_and_key(DIM, model_key)
            optim = optax.sgd(LR)
            opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
            if method != "delayed_mlmc":
                grad_per_level = [tree_zeros_like(model) for l in range(MAX_LEVEL + 1)]

            losses_inner = []
            pbar = trange(N_ITER, leave=False)
            for i in pbar:
                key_inner = jr.fold_in(key_outer, i)
                model_prev = model
                if method != "delayed_mlmc":
                    model, opt_state, grad_per_level = step_func(
                        model, opt_state, grad_per_level, key_inner, i
                    )
                else:
                    model, opt_state = step_func(model, opt_state, key_inner)
                loss = jnp.mean(batched_loss_baseline(model, jr.split(key_inner, BATCH_SIZE)))
                losses_inner.append(loss)
                pbar.set_description(desc="Step: {:>3d}, Loss: {:>5.2f}".format(i, float(loss)))
            losses_outer.append(jnp.stack(losses_inner))

        save_array(jnp.stack(losses_outer), save_path / f"losses_{method}.npy")
        losses_all.append(jnp.stack(losses_outer))

    save_array(jnp.stack(losses_all), save_path / f"losses_all.npy")


def examine_mlmc_decay() -> None:
    """Record the variance and smoothness per level during the optimization."""
    timestamp = get_timestamp()
    save_path = Path("./logs") / timestamp
    key, model_key = jr.split(KEY, 2)
    model = model_prev = DeepHedgingLoss.create_from_dim_and_key(DIM, model_key)
    optim = optax.sgd(LR)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    # @simple_filter_jit
    @eqx.filter_jit
    def step(
        model: DeepHedgingLoss, opt_state: optax.OptState, key: PRNGKeyArray
    ) -> Tuple[Float[Array, ""], DeepHedgingLoss, optax.OptState]:
        keys = jr.split(key, BATCH_SIZE)
        loss, grad = loss_and_grad(model, keys, 0)
        updates, opt_state = optim.update(grad, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    losses = []
    log_normalized_grad_diff_l2_norms(model, save_path / "before", key)
    log_grad_l2_norms(model, save_path / "before", key)

    pbar = trange(N_ITER + 1)
    for i in pbar:
        key = jr.fold_in(key, i)
        loss, model, opt_state = step(model, opt_state, key)
        losses.append(loss)
        pbar.set_description(desc="Step: {:>3d}, Loss: {:>5.2f}".format(i, loss))

    save_array(jnp.stack(losses), save_path / "losses.npy")

    log_normalized_grad_diff_l2_norms(model, save_path / "after", key)
    log_grad_l2_norms(model, save_path / "after", key)


if __name__ == "__main__":
    logging.getLogger("jax").setLevel(logging.INFO)
    jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]
    # jax_smi.initialise_tracking()
    # jax.experimental.compilation_cache.compilation_cache.initialize_cache(
    #     "./.compilation_cache"
    # )  # type: ignore[no-untyped-call]
    # jax.experimental.io_callback()
    with jax.disable_jit(False):
        # examine_mlmc_decay()
        run_deep_hedging()
