import math
import os
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional

import fire
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import tab10

plt.style.use("seaborn-whitegrid")

ArrayDict = Dict[str, List[Optional[np.ndarray]]]


def get_all_timestamps() -> List[str]:
    return [ts for ts in os.listdir("./logs") if ts.startswith("202") and len(ts) == 14]


def get_latest_timestamp() -> str:
    return max(get_all_timestamps())


def try_np_load(path: Path) -> Optional[np.ndarray]:
    try:
        return np.load(path)
    except FileNotFoundError:
        return None


def try_load_arrays(timestamps: List[str]) -> ArrayDict:
    array_dict: ArrayDict = defaultdict(list)
    for ts in timestamps:
        log_dir = Path("./logs") / ts
        array_dict["losses"] += [try_np_load(log_dir / "losses.npy")]
        array_dict["diff_norms_before"] += [
            try_np_load(log_dir / "before" / "normalized_grad_diff_l2_norms.npy")
        ]
        array_dict["norms_before"] += [try_np_load(log_dir / "before" / "grad_l2_norms.npy")]
        array_dict["diff_norms_after"] += [
            try_np_load(log_dir / "after" / "normalized_grad_diff_l2_norms.npy")
        ]
        array_dict["norms_after"] += [try_np_load(log_dir / "after" / "grad_l2_norms.npy")]
        array_dict["losses_dmlmc"] += [try_np_load(log_dir / "losses_delayed_mlmc.npy")]
        array_dict["times_dmlmc"] += [try_np_load(log_dir / "times_delayed_mlmc.npy")]
        array_dict["losses_mlmc"] += [try_np_load(log_dir / "losses_mlmc.npy")]
        array_dict["times_mlmc"] += [try_np_load(log_dir / "time_mlmc.npy")]
        array_dict["losses_base"] += [try_np_load(log_dir / "losses_baseline.npy")]
        array_dict["times_base"] += [try_np_load(log_dir / "times_baseline.npy")]
    return array_dict


def plot_variance_decay(array_dict: ArrayDict) -> None:
    norms = array_dict["norms_after"][0]
    assert isinstance(norms, np.ndarray)
    mean_per_level = (norms**2).mean(axis=1)
    std_per_level = (norms**2).std(axis=1) / norms.shape[1] ** 0.5
    std_per_level_log_trans = std_per_level / mean_per_level
    upper = mean_per_level * np.exp(std_per_level / mean_per_level)
    lower = mean_per_level / np.exp(std_per_level / mean_per_level)
    levels = range(norms.shape[0])
    line = plt.plot(levels, mean_per_level)[0]
    band = plt.fill_between(levels, upper, lower, alpha=0.3)
    O05 = plt.plot(levels, [mean_per_level[0] * 2 ** (-0.5 * l) for l in levels], c=tab10(7))[0]
    O1 = plt.plot(levels, [mean_per_level[0] * 2 ** (-l) for l in levels], c=tab10(7))[0]
    O2 = plt.plot(levels, [mean_per_level[0] * 2 ** (-2 * l) for l in levels], c=tab10(7))[0]
    plt.legend(
        [(line, band), (O05,), (O1,), (O2,)],
        [
            r"$\mathrm{E}\|\Delta\nabla_\ell \hat F\|^2$",
            r"$O(2^{-\ell/2})$",
            r"$O(2^{-\ell})$",
            r"$O(2^{-2\ell})$",
        ],
    )
    plt.xlabel(r"Level $\ell$")
    plt.ylabel(
        r"Trace of the second moment of the coupled gradient estimator $\Delta\nabla_\ell \hat F$"
    )
    plt.yscale("log")
    plt.savefig("./logs/variance_decay.pdf")


def plot_smoothness_decay(array_dict: ArrayDict) -> None:
    norms = array_dict["diff_norms_after"][0]
    assert isinstance(norms, np.ndarray)
    mean_per_level = norms.mean(axis=1)
    std_per_level = norms.std(axis=1) / norms.shape[1] ** 0.5
    std_per_level_log_trans = std_per_level / mean_per_level
    upper = mean_per_level * np.exp(std_per_level / mean_per_level)
    lower = mean_per_level / np.exp(std_per_level / mean_per_level)
    levels = range(norms.shape[0])
    line = plt.plot(levels, mean_per_level)[0]
    band = plt.fill_between(levels, upper, lower, alpha=0.3)
    O05 = plt.plot(levels, [mean_per_level[0] * 2 ** (-0.5 * l) for l in levels], c=tab10(7))[0]
    O1 = plt.plot(levels, [mean_per_level[0] * 2 ** (-l) for l in levels], c=tab10(7))[0]
    O2 = plt.plot(levels, [mean_per_level[0] * 2 ** (-2 * l) for l in levels], c=tab10(7))[0]
    smoothness_definition = r"$\mathrm{E}\left\|\frac{\Delta\nabla_\ell \hat F(x_{t+1}, \xi_{t+1}) - \Delta\nabla_\ell \hat F(x_t, \xi_t)}{x_{t+1} - x_t}\right\|$"
    plt.legend(
        [(line, band), (O05,), (O1,), (O2,)],
        [smoothness_definition, r"$O(2^{-\ell/2})$", r"$O(2^{-\ell})$", r"$O(2^{-\ell/2})$"],
    )
    plt.xlabel(r"Level $\ell$")
    plt.ylabel(r"Mean of per-step smoothness")
    plt.yscale("log")
    plt.savefig("./logs/smoothness_decay.pdf")


MAX_LEVEL = 6
LEVELS = [i for i in range(1 + MAX_LEVEL)]
BATCH_SIZE = 2**9
COST_RATE = 1.0
SMOOTHNESS_DECAY_RATE = 1.0
VARIANCE_DECAY_RATE = 1.8  # TODO
PERIOD_PER_LEVEL = [math.floor(2 ** (1 + SMOOTHNESS_DECAY_RATE * (level - 1))) for level in LEVELS]
COST_PER_LEVEL = [
    2**level
    * math.ceil(BATCH_SIZE / 2 ** (0.5 * (VARIANCE_DECAY_RATE + COST_RATE) * level))
    / BATCH_SIZE
    for level in LEVELS
]


def apply_smoothing(signal: np.ndarray) -> np.ndarray:
    filter_size = len(signal) // 5
    _filter = np.ones(filter_size)
    filtered = np.convolve(signal, _filter)[: len(signal)]
    filtered /= np.minimum(np.arange(1, len(signal) + 1), filter_size)
    return filtered


def plot_learning_curves(array_dict: ArrayDict) -> None:
    losses_base = np.concatenate(array_dict["losses_base"])  # type: ignore
    losses_mlmc = np.concatenate(array_dict["losses_mlmc"])  # type: ignore
    losses_dmlmc = np.concatenate(array_dict["losses_dmlmc"])  # type: ignore
    mean_base = np.mean(losses_base, axis=0)
    mean_mlmc = np.mean(losses_mlmc, axis=0)
    mean_dmlmc = np.mean(losses_dmlmc, axis=0)
    std_base = np.std(losses_base, ddof=1, axis=0)
    std_mlmc = np.std(losses_mlmc, ddof=1, axis=0)
    std_dmlmc = np.std(losses_dmlmc, ddof=1, axis=0)
    mean_base, mean_mlmc, mean_dmlmc, std_base, std_mlmc, std_dmlmc = map(
        apply_smoothing,
        [mean_base, mean_mlmc, mean_dmlmc, std_base, std_mlmc, std_dmlmc],
    )

    tot_cost_base = np.cumsum([2**MAX_LEVEL for step in range(losses_base.shape[1])])
    tot_cost_mlmc = np.cumsum([sum(COST_PER_LEVEL) for step in range(losses_mlmc.shape[1])])
    tot_cost_dmlmc = np.cumsum(
        [
            sum(COST_PER_LEVEL[l] for l in LEVELS if step % PERIOD_PER_LEVEL[l] == 0)
            for step in range(losses_dmlmc.shape[1])
        ]
    )

    para_cost_base = np.cumsum([2**MAX_LEVEL for step in range(losses_base.shape[1])])
    para_cost_mlmc = np.cumsum([2**MAX_LEVEL for step in range(losses_mlmc.shape[1])])
    para_cost_dmlmc = np.cumsum(
        [
            2 ** max(l for l in LEVELS if step % PERIOD_PER_LEVEL[l] == 0)
            for step in range(losses_dmlmc.shape[1])
        ]
    )

    def _plot(cost_base: np.ndarray, cost_mlmc: np.ndarray, cost_dmlmc: np.ndarray) -> None:
        line_base = plt.plot(cost_base, mean_base, c=tab10(0))[0]
        line_mlmc = plt.plot(cost_mlmc, mean_mlmc, c=tab10(1))[0]
        line_dmlmc = plt.plot(cost_dmlmc, mean_dmlmc, c=tab10(2))[0]

        band_base = plt.fill_between(
            cost_base, mean_base - std_base, mean_base + std_base, color=tab10(0), alpha=0.2
        )
        band_mlmc = plt.fill_between(
            cost_mlmc, mean_mlmc - std_mlmc, mean_mlmc + std_mlmc, color=tab10(1), alpha=0.2
        )
        band_dmlmc = plt.fill_between(
            cost_dmlmc, mean_dmlmc - std_dmlmc, mean_dmlmc + std_dmlmc, color=tab10(2), alpha=0.2
        )

        plt.xlim([0, min(max(cost_base), max(cost_mlmc), max(cost_dmlmc))])
        plt.ylabel("Losson")
        plt.xlabel("Cumurative parallel complexity")

        plt.legend(
            [(line_base, band_base), (line_mlmc, band_mlmc), (line_dmlmc, band_dmlmc)],
            ["baseline", "MLMC", "delayed MLMC"],
        )

    _plot(tot_cost_base, tot_cost_mlmc, tot_cost_dmlmc)
    plt.savefig("logs/learning_curve_total_complexity.pdf")
    _plot(para_cost_base, para_cost_mlmc, para_cost_dmlmc)
    plt.savefig("logs/learning_curve_parallel_complexity.pdf")


def main(timestamps: Optional[List[int]] = None) -> None:
    parsed_timestamps: List[str] = (
        [str(t) for t in timestamps] if timestamps else [get_latest_timestamp()]
    )

    parsed_timestamps = [ts for ts in get_all_timestamps() if ts >= "20230927"]

    array_dict = try_load_arrays(parsed_timestamps)
    array_dict = {k: list(filter(lambda x: x is not None, v)) for k, v in array_dict.items()}

    # plot_variance_decay(array_dict)
    # plot_smoothness_decay(array_dict)
    # plot_learning_curves(array_dict)


if __name__ == "__main__":
    fire.Fire(main)
