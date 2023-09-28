import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import fire
import matplotlib.pyplot as plt
import numpy as np

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


def plot_decays(array_dict: ArrayDict) -> None:
    norms = array_dict["norms_after"][0]
    mean_per_level = (norms**2).mean(axis=1)
    std_per_level = (norms**2).std(axis=1) / norms.shape[1] ** 0.5
    std_per_level_log_trans = std_per_level / mean_per_level
    upper = mean_per_level * np.exp(std_per_level / mean_per_level)
    lower = mean_per_level / np.exp(std_per_level / mean_per_level)
    levels = np.arange(norms.shape[0])
    plt.plot(levels, mean_per_level)
    plt.fill_between(levels, upper, lower, alpha=0.3)
    plt.xlabel("Level")
    plt.ylabel(r"Trace of the second moment of gradient $\mathrm{E}[\|\Delta\nabla_\ell \hat F\|$")


def main(timestamps: Optional[List[int]] = None) -> None:
    parsed_timestamps: List[str] = (
        [str(t) for t in timestamps] if timestamps else [get_latest_timestamp()]
    )

    parsed_timestamps = [ts for ts in get_all_timestamps() if ts >= "20230927"]

    array_dict = try_load_arrays(parsed_timestamps)
    array_dict = {k: list(filter(lambda x: x is not None, v)) for k, v in array_dict.items()}

    plot_decays(array_dict)
    breakpoint()


if __name__ == "__main__":
    fire.Fire(main)
