from functools import partial
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from diffrax import (AbstractBrownianPath, ControlTerm, ItoMilstein, MultiTerm,
                     ODETerm, SaveAt, StepTo, diffeqsolve)
from jaxtyping import Array, Float

t0, t1, n_steps, dim = 0, 1, 100, 1

batch_size, lr, n_iter = 8, 1e-3, 100

MU = 1.0
SIGMA = 1.0
STRIKE_PRICE = 1.5


class FixedBrownianMotion(AbstractBrownianPath):
    bm: Float[Array, "T+1 dim"] = eqx.field(static=True)
    t0: float = eqx.field(static=True)
    t1: float = eqx.field(static=True)
    n_steps: int = eqx.field(static=True)

    def evaluate(self, t0, t1=None, left=True) -> Float[Array, "dim"]:
        del left
        if t1 is not None:
            return self.evaluate(t1) - self.evaluate(t0)
        idx = (self.n_steps * (t0 - self.t0)).astype(int)
        return self.bm[idx]

    @classmethod
    def create_from_interval(cls, t0, t1, n_steps, dim, key, antithetic=False):
        rnorm = jax.random.normal(key, (n_steps, dim))
        bm = jnp.cumsum(
            ((t1 - t0) / n_steps) ** 0.5
            * jnp.concatenate([jnp.zeros((1, dim)), rnorm], axis=0),
            axis=0,
        )
        return cls(bm=bm, t0=t0, t1=t1, n_steps=n_steps)

    def to_antithetic_path(self):
        assert self.n_steps % 2 == 0
        bm = self.bm.at[1::2].set(
            self.bm[::2][:-1] + self.bm[::2][1:] - self.bm[1:][::2]
        )
        return FixedBrownianMotion(bm=bm, t0=self.t0, t1=self.t1, n_steps=self.n_steps)


class HoldingStrategy(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, key):
        self.mlp = eqx.nn.MLP(
            in_size=2,
            out_size=1,
            width_size=32,
            depth=4,
            activation=jax.nn.silu,
            final_activation=jax.nn.sigmoid,
            key=key,
        )

    @jax.named_scope("HoldingStrategy")
    def __call__(self, t, y):
        return self.mlp(jnp.concatenate([t[None], y]))


def BSDrift(t, y: Float[Array, "dim"]) -> Float[Array, "dim"]:
    return MU * y


def BSDiffusion(t, y: Float[Array, "dim"]) -> Float[Array, "dim dim"]:
    return jnp.diag(SIGMA * y)


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
    def create_from_asset_process(cls, key, diffusion=None, drift=None, dim=1):
        assert dim == 1
        h = HoldingStrategy(key)
        w = jnp.array(0.0)
        s_drift = drift or BSDrift
        s_diffusion = diffusion or BSDiffusion
        return cls(h=h, w=w, s_drift=s_drift, s_diffusion=s_diffusion, dim=dim)

    def drift(self, t, y: Float[Array, "dim+1"], args) -> Float[Array, "dim+1"]:
        # We can only handle cases where dim = 1.
        s = y[:dim]
        dhdt = jax.jacobian(self.h, argnums=0)(t, s)
        dhds = jax.jacobian(self.h, argnums=1)(t, s)[:, 0]
        dhdss = jax.hessian(self.h, argnums=1)(t, s)[:, 0, 0]
        s_drift = self.s_drift(t, s)
        s_diffusion = jnp.diag(self.s_diffusion(t, s))
        h_drift = dhdt + dhds * s_drift + dhdss * s_diffusion**2 / 2
        return jnp.concatenate([s_drift, s * h_drift], axis=0)

    def diffusion(self, t, y: Float[Array, "dim+1"], args) -> Float[Array, "dim+1 dim"]:
        # We can only handle cases where dim = 1.
        s = y[:dim]
        dhds = jax.jacobian(self.h, argnums=1)(t, s)[:, 0]
        s_diffusion = self.s_diffusion(t, s)
        h_diffusion = dhds * s_diffusion
        return jnp.concatenate([s_diffusion, s * h_diffusion], axis=0)

    @jax.named_scope("DeepHedgingLoss")
    def __call__(self, bm, ts: Float[Array, " T+1"]) -> Float[Array, ""]:
        terms = MultiTerm(ODETerm(self.drift), ControlTerm(self.diffusion, bm))
        solver = ItoMilstein()
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
        J = self.w + loss(z - hedging_pnl - self.w)
        return J


def run_deep_hedging():
    key, model_key = jax.random.split(jax.random.PRNGKey(0), 2)
    model = DeepHedgingLoss.create_from_asset_process(model_key)
    optim = optax.sgd(lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    @partial(jax.vmap, in_axes=(None, 0))
    def batched_loss(model, key):
        bm = FixedBrownianMotion.create_from_interval(t0, t1, n_steps, dim, key)
        ts = (t1 - t0) * jnp.arange(0, n_steps + 1) / n_steps + t0
        return model(bm, ts)

    @eqx.filter_value_and_grad
    def loss_and_grad(hedging_loss, keys):
        return jnp.mean(batched_loss(hedging_loss, keys))

    @eqx.filter_jit
    def step(model, opt_state, key):
        keys = jax.random.split(key, batch_size)
        loss, grad = loss_and_grad(model, keys)
        updates, opt_state = optim.update(grad, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    ret = batched_loss(model, jax.random.split(key, batch_size))
    breakpoint()

    for i in range(n_iter):
        key = jax.random.fold_in(key, i)
        loss, model, opt_state = step(model, opt_state, key)
        print(loss)
    return ret


def test():
    key = jax.random.PRNGKey(0)
    brownian_motion = FixedBrownianMotion.create_from_interval(
        t0, t1, n_steps, dim, key
    )
    brownian_motion_a = brownian_motion.to_antithetic_path()
    ts_fine = (t1 - t0) * jnp.arange(0, n_steps + 1) / n_steps + t0
    ts_coarse = ts_fine[::2]

    drift = lambda t, y, args: y  # noqa
    diffusion = lambda t, y, args: jnp.ones((1,))  # noqa

    terms = MultiTerm(ODETerm(drift), ControlTerm(diffusion, brownian_motion))
    terms_a = MultiTerm(ODETerm(drift), ControlTerm(diffusion, brownian_motion_a))
    solver = ItoMilstein()
    saveat = SaveAt(dense=True)

    sol = diffeqsolve(
        terms,
        solver,
        t0,
        t1,
        dt0=None,
        y0=1.0,
        saveat=saveat,
        stepsize_controller=StepTo(ts_coarse),
    )
    sol_f = diffeqsolve(
        terms,
        solver,
        t0,
        t1,
        dt0=None,
        y0=1.0,
        saveat=saveat,
        stepsize_controller=StepTo(ts_fine),
    )
    sol_fa = diffeqsolve(
        terms_a,
        solver,
        t0,
        t1,
        dt0=None,
        y0=1.0,
        saveat=saveat,
        stepsize_controller=StepTo(ts_fine),
    )

    print(
        sol.evaluate(0.5),
        (sol_f.evaluate(0.5) + sol_fa.evaluate(0.5)) / 2,
        sol_f.evaluate(0.5),
        sol_fa.evaluate(0.5),
    )


if __name__ == "__main__":
    # jax.config.update("jax_enable_x64", True)
    # jax.config.update("jax_platform_name", "cpu")
    # jax.experimental.io_callback()
    with jax.disable_jit(False):
        # test()
        run_deep_hedging()
