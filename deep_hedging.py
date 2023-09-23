from functools import partial
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from diffrax import (AbstractBrownianPath, ControlTerm, ItoMilstein, MultiTerm,
                     ODETerm, SaveAt, StepTo, diffeqsolve)
from jaxtyping import Array, Float
from tqdm import tqdm

t0, t1, n_steps, dim = 0, 1, 10, 1

batch_size, lr, n_iter, max_level = 2**12, 1e-2, 100, 8

MU = 1.0
SIGMA = 1.0
STRIKE_PRICE = jnp.exp(MU)


class FixedBrownianMotion(AbstractBrownianPath):
    bm: Float[
        Array, "n_steps+1 dim"
    ]  #  = eqx.field(static=True)  # this causes leaked tracer error
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
    def __call__(self, bm, ts: Float[Array, " n_steps+1"]) -> Float[Array, ""]:
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
        w = 10 * self.w  # rescale to match learning rate with NN
        J = w + loss(z - hedging_pnl - w)
        return J


def run_deep_hedging():
    key, model_key = jax.random.split(jax.random.PRNGKey(0), 2)
    model = DeepHedgingLoss.create_from_asset_process(model_key)
    optim = optax.sgd(lr)
    # optim = optax.adadelta(lr)
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

    losses = []
    pbar = tqdm(range(n_iter))
    for i in pbar:
        key = jax.random.fold_in(key, i)
        loss, model, opt_state = step(model, opt_state, key)
        losses.append(loss)
        pbar.set_description(desc="Step: {:>3d}, Loss: {:>5.2f}".format(i, loss))


def run_mlmc_deep_hedging():
    key, model_key = jax.random.split(jax.random.PRNGKey(0), 2)
    model = DeepHedgingLoss.create_from_asset_process(model_key)
    optim = optax.sgd(lr)
    # optim = optax.adadelta(lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    @partial(jax.vmap, in_axes=(None, 0, None))
    def batched_loss(model, key, level):
        bm = FixedBrownianMotion.create_from_interval(
            t0, t1, n_steps * 2**level, dim, key
        )
        abm = bm.to_antithetic_path()
        ts = (t1 - t0) * jnp.arange(0, n_steps * 2**level + 1) / (
            n_steps * 2**level
        ) + t0
        ts_coarse = ts[::2]
        if level == 0:
            return model(bm, ts)
        else:
            return (model(bm, ts) + model(abm, ts)) / 2 - model(bm, ts_coarse)

    @eqx.filter_value_and_grad
    def loss_and_grad(model, keys, level):
        return jnp.mean(batched_loss(model, keys, level))

    @eqx.filter_jit
    @partial(jax.vmap, in_axes=(None, 0, None))
    def grad_l2_norm(model, keys, level):
        l, g = loss_and_grad(model, keys, level)  # noqa
        squared_sum = sum(jnp.sum(p**2) for p in jax.tree_util.tree_leaves(g))
        return squared_sum**0.5

    @eqx.filter_jit
    def step(model, opt_state, key):
        keys = jax.random.split(key, batch_size)
        loss, grad = loss_and_grad(model, keys, 0)
        updates, opt_state = optim.update(grad, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    keys_2d = jax.random.split(key, (2**8, 1))
    l2_norms = []
    for l in range(max_level):
        l2_norms.append(grad_l2_norm(model, keys_2d, l))
    # l2_norm = jnp.mean(l2_norm)

    losses = []
    pbar = tqdm(range(n_iter))
    for i in pbar:
        key = jax.random.fold_in(key, i)
        loss, model, opt_state = step(model, opt_state, key)
        losses.append(loss)
        pbar.set_description(desc="Step: {:>3d}, Loss: {:>5.2f}".format(i, loss))

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
    from diffrax import Euler

    solver = Euler()
    # solver = ItoMilstein()
    saveat = SaveAt(t1=True)

    from diffrax import (BacksolveAdjoint, ConstantStepSize, DirectAdjoint,
                         VirtualBrownianTree)

    brownian_motion = VirtualBrownianTree(t0, t1, tol=0.1, shape=(1,), key=key)
    terms = MultiTerm(ODETerm(drift), ControlTerm(diffusion, brownian_motion))
    mk = eqx.filter_make_jaxpr(diffeqsolve)

    f = lambda: mk(
        ODETerm(drift),
        solver,
        t0,
        t1,
        dt0=None,
        y0=1.0,
        saveat=saveat,
        stepsize_controller=StepTo(ts_coarse),
    )
    ff = lambda: mk(
        terms,
        solver,
        t0,
        t1,
        dt0=None,
        y0=1.0,
        saveat=saveat,
        stepsize_controller=StepTo(ts_coarse),
    )

    g = lambda: mk(ODETerm(drift), solver, t0, t1, dt0=0.2, y0=1.0, saveat=saveat)
    gg = lambda: mk(
        terms,
        solver,
        t0,
        t1,
        dt0=0.2,
        y0=1.0,
        saveat=saveat,
        stepsize_controller=ConstantStepSize(compile_steps=True),
    )

    h = lambda: mk(
        ODETerm(drift),
        solver,
        t0,
        t1,
        dt0=0.2,
        y0=1.0,
        saveat=saveat,
        adjoint=BacksolveAdjoint(),
    )
    hh = lambda: mk(
        terms,
        solver,
        t0,
        t1,
        dt0=0.2,
        y0=1.0,
        saveat=saveat,
        adjoint=BacksolveAdjoint(),
    )

    k = lambda: mk(
        ODETerm(drift),
        solver,
        t0,
        t1,
        dt0=0.2,
        y0=1.0,
        saveat=saveat,
        adjoint=DirectAdjoint(),
    )
    kk = lambda: mk(
        terms, solver, t0, t1, dt0=0.2, y0=1.0, saveat=saveat, adjoint=DirectAdjoint()
    )
    breakpoint()

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
    with jax.disable_jit(True):
        test()
        run_mlmc_deep_hedging()
