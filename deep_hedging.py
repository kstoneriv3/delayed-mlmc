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

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import mlflow
import numpy as np
import optax
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
# jax.config.update("jax_platform_name", "cpu")  # type: ignore[no-untyped-call]


T0, T1, N_STEPS_LEVEL0, DIM = 0, 1, 10, 1

BATCH_SIZE, LR, N_ITER, MAX_LEVEL = 2**11, 1e-3, 1000, 7
# BATCH_SIZE, LR, N_ITER, MAX_LEVEL = 2**9, 1e-3, 800, 6
# BATCH_SIZE, LR, N_ITER, MAX_LEVEL = 2**5, 1e-2, 20, 1

KEY = jr.PRNGKey(10)
MU = 1.0
SIGMA = 1.0
STRIKE_PRICE = 3.0

N_REPEAT_EXPERIMENT = 1
COST_RATE = 1
VARIANCE_DECAY_RATE = 1.8
SMOOTHNESS_DECAY_RATE = 1.0

LEVELS = [l for l in range(1 + MAX_LEVEL)]
PERIOD_PER_LEVEL = [math.floor(2 ** (1 + SMOOTHNESS_DECAY_RATE * (l - 1))) for l in LEVELS]
BATCH_SIZE_PER_LEVEL = [
    math.ceil(BATCH_SIZE / 2 ** (0.5 * (VARIANCE_DECAY_RATE + COST_RATE) * l)) for l in LEVELS
]
VARIANCE_PER_LEVEL = [
    2 ** (-VARIANCE_DECAY_RATE * l) * BATCH_SIZE / b for b, l in zip(BATCH_SIZE_PER_LEVEL, LEVELS)
]
TOTAL_MLMC_VARIANCE = sum(VARIANCE_PER_LEVEL)

VALIDATION_CYCLE = 2**MAX_LEVEL


@contextmanager  # type: ignore[arg-type]
def nullcontext(*args: Any, **kwargs: Any) -> None:  # type: ignore
    yield


def nullfunc(*args: Any, **kwargs: Any) -> None:
    pass


def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d%H%M%S")


def save_array(array: Array, path: Path) -> None:
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
            depth=2,
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
        keys = jr.split(key, 2)
        h = HoldingStrategy(dim, keys[0])
        w = jnp.mean(
            jnp.maximum(
                jnp.exp(MU - SIGMA**2 / 2 + SIGMA * jr.normal(keys[1], (10**6,)))
                - STRIKE_PRICE,
                0,
            )
        )
        return cls(h=h, w=w, s_drift=BlackScholesDrift, s_diffusion=BlackScholesDiffusion, dim=dim)

    def drift(self, t: Scalar, y: Float[Array, "dim+1"], args: Any) -> Float[Array, "dim+1"]:
        # We can only handle cases where dim = 1.
        s = y[: self.dim]
        s_drift = self.s_drift(t, s)
        h_drift = s_drift * self.h(t, s)
        return jnp.concatenate([s_drift, h_drift], axis=0)

    def diffusion(
        self, t: Scalar, y: Float[Array, "dim+1"], args: Any
    ) -> Float[Array, "dim+1 dim"]:
        # We can only handle cases where dim = 1.
        s = y[: self.dim]
        s_diffusion = self.s_diffusion(t, s)
        h_diffusion = s_diffusion * self.h(t, s)[None]
        return jnp.concatenate([s_diffusion, h_diffusion], axis=0)

    @jax.named_scope("DeepHedgingLoss")
    def __call__(
        self, bm: AbstractBrownianPath, ts: Float[Array, " n_steps+1"]
    ) -> Float[Array, ""]:
        terms = MultiTerm(ODETerm(self.drift), ControlTerm(self.diffusion, bm))
        solver = ItoMilstein()
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
        w = self.w  # rescale to match learning rate with NN
        J = (z - hedging_pnl - w) ** 2  # type: ignore[no-untyped-call]
        return J


def save_model_weight(model: DeepHedgingLoss, path: Path) -> None:
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
    ts = (T1 - T0) * jnp.arange(n_steps + 1) / n_steps + T0
    ts_coarse = ts[::2]
    if level == 0:
        return model(bm, ts)
    else:
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
    norms_outer = [None] * (MAX_LEVEL + 1)
    for level in tqdm(reversed(LEVELS), desc=f"Evaluating variance"):
        norms_inner = []
        for i in trange(2**8, desc=f"Level {level}", leave=False):
            keys_2d = jr.split(jr.fold_in(key, i), (2**8, 1))
            norms_inner.append(grad_l2_norm(model, keys_2d, level))
        norms_outer[level] = jnp.concatenate(norms_inner)
        clear_all_jit_cache()
        jax.lib.xla_bridge.get_backend().defragment()
    norms = jnp.stack(norms_outer)  # type: ignore[arg-type]
    fp = save_path / "grad_l2_norms.npy"
    save_array(norms, fp)
    mlflow.log_artifacts(fp.parent.absolute().as_posix(), fp.name)


@eqx.filter_jit
def perturb_model(model: DeepHedgingLoss, key: PRNGKeyArray) -> DeepHedgingLoss:
    keys = jr.split(key, 2**6)
    loss, grad = loss_and_grad(model, keys, 0)
    update = jtu.tree_map(lambda x: -1e-4 * x, grad)
    return eqx.apply_updates(model, update)


def log_normalized_grad_diff_l2_norms(
    model: DeepHedgingLoss,
    save_path: Path,
    key: PRNGKeyArray,
) -> None:
    norms_outer = [None] * (MAX_LEVEL + 1)
    for level in tqdm(reversed(LEVELS), desc=f"Evaluating smoothness"):
        norms_inner = []
        for i in trange(2**6, desc=f"Level {level}", leave=False):
            key_loop = jr.fold_in(key, i)
            model_perturbed = perturb_model(model, key_loop)
            keys = jr.split(key_loop, (2**9,))
            norms_inner.append(
                grad_diff_l2_norm(model, model_perturbed, keys, level)
                / param_diff_l2_norm(model, model_perturbed)
            )
        norms_outer[level] = jnp.stack(norms_inner)
        clear_all_jit_cache()
        jax.lib.xla_bridge.get_backend().defragment()
    norms = jnp.stack(norms_outer)  # type: ignore[arg-type]
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
    keys = jr.split(key, math.ceil(BATCH_SIZE / TOTAL_MLMC_VARIANCE))  # To match variance with MLMC
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
    grad_sum = tree_sum(grad_per_level)
    updates, opt_state = optim.update(grad_sum, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state


def step_mlmc(
    model: DeepHedgingLoss,
    optim: optax.GradientTransformation,
    opt_state: optax.OptState,
    key: PRNGKeyArray,
) -> Tuple[DeepHedgingLoss, optax.OptState]:
    grad_per_level = [None] * (MAX_LEVEL + 1)
    for level in reversed(LEVELS):
        batch_size = math.ceil(BATCH_SIZE / 2 ** (0.5 * (VARIANCE_DECAY_RATE + COST_RATE) * level))
        keys = jr.split(jr.fold_in(key, level), batch_size)
        loss, grad = loss_and_grad(model, keys, level)
        grad_per_level[level] = grad
    model, opt_state = _step_from_grad_per_level(model, optim, opt_state, grad_per_level)
    return model, opt_state


def step_delayed_mlmc(
    model: DeepHedgingLoss,
    optim: optax.GradientTransformation,
    opt_state: optax.OptState,
    grad_per_level: List[DeepHedgingLoss],
    key: PRNGKeyArray,
    step: int,
) -> Tuple[DeepHedgingLoss, optax.OptState, List[DeepHedgingLoss]]:
    keys = jr.split(key, BATCH_SIZE)
    for level in reversed(LEVELS):
        if step % PERIOD_PER_LEVEL[level] != 0:
            continue
        batch_size = math.ceil(BATCH_SIZE / 2 ** (0.5 * (VARIANCE_DECAY_RATE + COST_RATE) * level))
        keys = jr.split(jr.fold_in(key, level), batch_size)
        loss, grad = loss_and_grad(model, keys, level)
        grad_per_level[level] = grad

    model, opt_state = _step_from_grad_per_level(model, optim, opt_state, grad_per_level)
    return model, opt_state, grad_per_level


def run_deep_hedging() -> None:
    """Run deep hedging N_REPEAT_EXPERIMENT times for baseline, mlmc, and delayed-mlmc method."""

    mlflow.set_experiment("run_deep_hedging")
    timestamp = get_timestamp()
    save_path = Path("./logs") / timestamp

    # for method in tqdm(["baseline"]):
    # for method in tqdm(["mlmc"]):
    # for method in tqdm(["delayed_mlmc"]):
    # for method in tqdm(["delayed_mlmc", "mlmc", "baseline"]):
    for method in tqdm(["delayed_mlmc"]):
        losses_outer = []
        times_outer = []
        for n in trange(N_REPEAT_EXPERIMENT, desc=f"Using {method}", leave=False):
            key_outer, model_key = jr.split(jr.fold_in(KEY, n), 2)
            model = model_prev = DeepHedgingLoss.create_from_dim_and_key(DIM, model_key)
            models = [model, model_prev]

            optim = optax.sgd(LR)
            opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
            if method == "delayed_mlmc":
                grad_per_level = [tree_zeros_like(eqx.filter(model, eqx.is_array)) for _ in LEVELS]

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
                            print(model.w)
                        case "mlmc":
                            model, opt_state = step_mlmc(model, optim, opt_state, key_inner)
                        case "delayed_mlmc":
                            model, opt_state, grad_per_level = step_delayed_mlmc(
                                model, optim, opt_state, grad_per_level, key_inner, i
                            )
                    keys_loss = jr.split(jr.fold_in(KEY, i % VALIDATION_CYCLE), BATCH_SIZE)
                    loss = jnp.mean(batched_loss_baseline(model, keys_loss, MAX_LEVEL))
                    losses_inner.append(loss)
                    times_inner.append(time.time())
                    pbar.set_description(desc="Step: {:>3d}, Loss: {:>5.2f}".format(i, float(loss)))
                    mlflow.log_metric("loss", float(loss), i)
                losses_outer.append(jnp.stack(losses_inner))
                times_outer.append(jnp.stack(losses_inner))
                save_model_weight(model, save_path / f"model_{method}_{i}.eqx")

        save_array(jnp.stack(losses_outer), save_path / f"losses_{method}.npy")
        save_array(jnp.stack(times_outer), save_path / f"times_{method}.npy")


def examine_mlmc_decay() -> None:
    """Record the variance and smoothness per level during the optimization."""
    mlflow.set_experiment("examine_mlmc_decay")

    timestamp = get_timestamp()

    save_path = Path("./logs") / timestamp
    key, model_key = jr.split(KEY, 2)
    model = model_prev = DeepHedgingLoss.create_from_dim_and_key(DIM, model_key)

    with mlflow.start_run():
        global_params = {
            k: v
            for k, v in globals().items()
            if k.upper() == k and isinstance(v, (int, float, str))
        }
        global_params.update({"KEY": int(KEY[1])})  # type: ignore
        mlflow.log_params(global_params)

        log_normalized_grad_diff_l2_norms(model, save_path / "before", key)
        log_grad_l2_norms(model, save_path / "before", key)

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
    with jax.disable_jit(False):
        examine_mlmc_decay()
        exit()
        for k in range(5, 10):
            KEY = jr.PRNGKey(20 + k)
            run_deep_hedging()
