from typing import Callable, Union

import equinox as eqx
import jax
import jax.numpy as jnp
from diffrax import (AbstractBrownianPath, ControlTerm, ItoMilstein, MultiTerm,
                     ODETerm, SaveAt, StepTo, diffeqsolve)
from jaxtyping import Array, Float

# jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

# jax.experimental.io_callback()


t0, t1, n_steps, dim = 0, 1, 1000, 1

MU = 1.0
SIGMA = 1.0


class FixedBrownianMotion(AbstractBrownianPath):
    bm: Float[Array, "T dim"] = eqx.field(static=False)
    t0: Union[float, jnp.ndarray]
    t1: Union[float, jnp.ndarray]
    n_steps: Union[int, jnp.ndarray]

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


class Holding(eqx.Module):
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

    @jax.named_scope("Holding")
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

    h: eqx.Module  # holding function h(t, s) that takes values in [0, 1]
    s_drift: Callable
    s_diffusion: Callable
    dim: Union[int, jnp.ndarray]

    @classmethod
    def create_from_asset_process(cls, key, diffusion=None, drift=None, dim=1):
        h = Holding(key)
        s_drift = drift or BSDrift
        s_diffusion = diffusion or BSDiffusion
        return cls(h=h, s_drift=s_drift, s_diffusion=s_diffusion, dim=dim)

    def drift(self, t, y: Float[Array, "dim*2"], args) -> Float[Array, "dim*2"]:
        # We can only handle cases where dim = 1.
        s = y[:dim]
        dhdt = jax.jacobian(self.h, argnums=0)(t, s)
        dhds = jax.jacobian(self.h, argnums=1)(t, s)[:, 0]
        dhdss = jax.hessian(self.h, argnums=1)(t, s)[:, 0, 0]
        s_drift = self.s_drift(t, s)
        s_diffusion = jnp.diag(self.s_diffusion(t, s))
        h_drift = dhdt + dhds * s_drift + dhdss * s_diffusion**2 / 2
        return jnp.concatenate(
            [
                s_drift,
                # s * h_drift,
                h_drift,
            ],
            axis=0,
        )

    def diffusion(self, t, y: Float[Array, "dim*2"], args) -> Float[Array, "dim*2 dim"]:
        # We can only handle cases where dim = 1.
        s = y[:dim]
        dhds = jax.jacobian(self.h, argnums=1)(t, s)[:, 0]
        s_diffusion = self.s_diffusion(t, s)
        h_diffusion = dhds * s_diffusion
        return jnp.concatenate(
            [
                s_diffusion,
                # s * h_diffusion,
                h_diffusion,
            ],
            axis=0,
        )

    @jax.named_scope("DeepHedgingLoss")
    def __call__(self, key):
        brownian_motion = FixedBrownianMotion.create_from_interval(
            t0, t1, n_steps, dim, key
        )
        ts_fine = (t1 - t0) * jnp.arange(0, n_steps + 1) / n_steps + t0
        terms = MultiTerm(
            ODETerm(self.drift), ControlTerm(self.diffusion, brownian_motion)
        )
        solver = ItoMilstein()
        saveat = SaveAt(t1=True)
        stepto = StepTo(ts_fine)
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
        breakpoint()
        return 0


def run_deep_hedging():
    keys = jax.random.split(jax.random.PRNGKey(0), 10)
    loss = DeepHedgingLoss.create_from_asset_process(keys[0])
    loss(keys[1])


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
    with jax.disable_jit(False):
        # test()
        run_deep_hedging()
