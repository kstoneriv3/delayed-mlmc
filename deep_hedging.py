from typing import Union

import equinox as eqx
import jax
import jax.numpy as jnp
from diffrax import (AbstractBrownianPath, ControlTerm, ItoMilstein, MultiTerm,
                     ODETerm, SaveAt, StepTo, diffeqsolve)

# jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

# jax.experimental.io_callback()


keys = jax.random.split(jax.random.PRNGKey(0), 10)

t0, t1, n_steps = 0, 1, 1000

mu = 1.0
sigma = 1.0


class FixedBrownianMotion(AbstractBrownianPath):
    bm: jax.Array = eqx.field(static=False)
    t0: Union[float, jnp.ndarray]
    t1: Union[float, jnp.ndarray]
    n_steps: Union[int, jnp.ndarray]

    def __init__(self, bm, t0, t1, n_steps):
        self.bm = bm
        self.t0 = t0
        self.t1 = t1
        self.n_steps = n_steps

    def evaluate(self, t0, t1=None, left=True):
        del left
        if t1 is not None:
            return self.evaluate(t1) - self.evaluate(t0)
        idx = (self.n_steps * (t0 - self.t0)).astype(int)
        return self.bm[idx]

    @classmethod
    def create_from_interval(cls, t0, t1, n_steps, key, antithetic=False):
        rnorm = jax.random.normal(key, (n_steps,))
        bm = jnp.cumsum(
            ((t1 - t0) / n_steps) ** 0.5
            * jnp.concatenate([jnp.zeros((1,)), rnorm], axis=0)
        )
        if antithetic:
            assert n_steps % 2 == 0
            bm = bm.at[1::2].set(bm[::2][:-1] + bm[::2][1:] - bm[1:][::2])
        return cls(bm=bm, t0=t0, t1=t1, n_steps=n_steps)


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
    def __call__(self, t, y, args):
        return self.mlp(jnp.concatenate([t[None], y]))


class DeepHedging(eqx.Module):
    holding: eqx.Module

    def __init__(self, key):
        keys = key.split()
        self.holding = Holding(keys[0])

    @jax.named_scope("DeepHedging")
    def __call__(self, key):
        return 0


def test():

    brownian_motion = FixedBrownianMotion.create_from_interval(t0, t1, n_steps, keys[0])
    brownian_motion_a = FixedBrownianMotion.create_from_interval(
        t0, t1, n_steps, keys[0], antithetic=True
    )
    ts_fine = (t1 - t0) * jnp.arange(0, n_steps + 1) / n_steps + t0
    ts_coarse = ts_fine[::2]

    drift = lambda t, y, args: y  # noqa
    diffusion = lambda t, y, args: 1.0  # noqa

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
        test()
