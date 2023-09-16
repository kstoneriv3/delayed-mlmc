import equinox as eqx
import jax
import jax.numpy as jnp
from diffrax import (AbstractBrownianPath, ControlTerm, ItoMilstein, MultiTerm,
                     ODETerm, SaveAt, StepTo, diffeqsolve)

# jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

jax.disable_jit()

keys = jax.random.split(jax.random.PRNGKey(0), 10)

t0, t1 = 1, 3


rnorm = jax.random.normal(keys[0], (1000,))
BM = jnp.cumsum(
    1000 ** (-0.5) * jnp.concatenate([jnp.zeros((1,)), rnorm, -rnorm], axis=0)
)
ABM = BM.at[1::2].set(BM[::2][:-1] + BM[::2][1:] - BM[1:][::2])


class FixedBrownianMotion(AbstractBrownianPath):
    bm: jax.Array = eqx.field(static=False)

    def __init__(self, bm):
        self.bm = bm

    @property
    def t0(self):
        return 1

    @property
    def t1(self):
        return 3

    def evaluate(self, t0, t1=None, left=True):
        del left
        if t1 is not None:
            return self.evaluate(t1) - self.evaluate(t0)
        idx = (1000 * (t0 - self.t0)).astype(int)
        return self.bm[idx]


drift = lambda t, y, args: y  # noqa
diffusion = lambda t, y, args: 1.0  # noqa
brownian_motion = FixedBrownianMotion(BM)
brownian_motion_a = FixedBrownianMotion(ABM)

breakpoint()

terms = MultiTerm(ODETerm(drift), ControlTerm(diffusion, brownian_motion))
terms_a = MultiTerm(ODETerm(drift), ControlTerm(diffusion, brownian_motion_a))
solver = ItoMilstein()
saveat = SaveAt(dense=True)

ts_coarse = jnp.arange(0, 1001) / 500 + 1.0
ts_fine = jnp.arange(0, 2001) / 1000 + 1.0

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

print(sol.evaluate(1.1))

print(1 + 0.1 + BM[100])


print(
    sol.evaluate(2.5),
    (sol_f.evaluate(2.5) + sol_fa.evaluate(2.5)) / 2,
    sol_f.evaluate(2.5),
    sol_fa.evaluate(2.5),
)
