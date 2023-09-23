from functools import partial
from typing import Any, Callable, Optional, Tuple, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
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
from jaxtyping import Array, Float, PRNGKeyArray
from tqdm import tqdm, trange

t0, t1, n_steps, dim = 0, 1, 10, 1

batch_size, lr, n_iter, max_level = 2**11, 1e-2, 1000, 7

KEY = jr.PRNGKey(0)
MU = 1.0
SIGMA = 1.0
STRIKE_PRICE = jnp.exp(MU)


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
        s = y[:dim]
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
        s = y[:dim]
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
            t0,
            t1,
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
def batched_loss(model: DeepHedgingLoss, key: PRNGKeyArray, level: Int) -> Float[Array, ""]:
    # bm = VirtualBrownianTree(t0, t1, tol=0.001, shape=(dim,), key=key)
    bm = FixedBrownianMotion.create_from_interval(t0, t1, n_steps * 2**level, dim, key)
    # abm = bm.to_antithetic_path()
    ts = (t1 - t0) * jnp.arange(0, n_steps * 2**level + 1) / (n_steps * 2**level) + t0
    ts_coarse = ts[::2]
    if level == 0:
        return model(bm, ts)
    else:
        # return (model(bm, ts) + model(abm, ts)) / 2 - model(bm, ts_coarse)
        return model(bm, ts) - model(bm, ts_coarse)


@eqx.filter_value_and_grad
def loss_and_grad(model: DeepHedgingLoss, keys: PRNGKeyArray, level: Int) -> Float[Array, ""]:
    return jnp.mean(batched_loss(model, keys, level))


@eqx.filter_jit
@partial(jax.vmap, in_axes=(None, 0, None))
def grad_l2_norm(model: DeepHedgingLoss, keys: PRNGKeyArray, level: Int) -> Float[Array, ""]:
    l, g = loss_and_grad(model, keys, level)  # noqa
    squared_sum = sum(jnp.sum(p**2) for p in jax.tree_util.tree_leaves(g))
    return squared_sum**0.5


def sample_grad_l2_norms(
    model: DeepHedgingLoss, key: PRNGKeyArray, max_level: Int
) -> Float[Array, "max_level 2**13"]:
    keys_3d = jr.split(key, (2**5, 2**8, 1))
    l2_norms = []
    for l in trange(max_level + 1, desc=f"Calculating variance decay"):
        list_l2_norm = []
        for keys_2d in tqdm(keys_3d, desc=f"Evaluating variance of level {l}", leave=False):
            list_l2_norm.append(grad_l2_norm(model, keys_2d, l))
        l2_norms.append(jnp.concatenate(list_l2_norm))
    return jnp.stack(l2_norms)


def run_mlmc_deep_hedging() -> None:
    key, model_key = jr.split(KEY, 2)
    model = DeepHedgingLoss.create_from_dim_and_key(dim, model_key)
    optim = optax.sgd(lr)
    # optim = optax.adadelta(lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def step(
        model: DeepHedgingLoss, opt_state: optax.OptState, key: PRNGKeyArray
    ) -> Tuple[Float[Array, ""], DeepHedgingLoss, optax.OptState]:
        keys = jr.split(key, batch_size)
        loss, grad = loss_and_grad(model, keys, 0)
        updates, opt_state = optim.update(grad, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    grad_l2_norms_before = sample_grad_l2_norms(model, key, max_level)

    # breakpoint()

    losses = []
    pbar = trange(n_iter)
    for i in pbar:
        key = jr.fold_in(key, i)
        loss, model, opt_state = step(model, opt_state, key)
        losses.append(loss)
        pbar.set_description(desc="Step: {:>3d}, Loss: {:>5.2f}".format(i, loss))


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]
    # jax.config.update("jax_platform_name", "cpu")  # type: ignore[no-untyped-call]
    # jax.experimental.io_callback()
    with jax.disable_jit(False):
        run_mlmc_deep_hedging()
