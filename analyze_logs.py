import os
from pathlib import Path
from typing import Optional

import fire
import matplotlib.pyplot as plt
import numpy as np


def get_latest_timestamp() -> str:
    timestamps = [ts for ts in os.listdir("./logs") if ts.startswith("202") and len(ts) == 14]
    return max(timestamps)


def try_np_load(path: Path) -> Optional[np.ndarray]:
    try:
        return np.load(path)
    except FileNotFoundError:
        return None


def main(timestamp: Optional[str] = None) -> None:
    timestamp = str(timestamp or get_latest_timestamp())
    log_dir = Path("./logs") / timestamp
    losses = try_np_load(log_dir / "losses.npy")
    diff_norms_before = try_np_load(log_dir / "before" / "normalized_grad_diff_l2_norms.npy")
    norms_before = try_np_load(log_dir / "before" / "grad_l2_norms.npy")
    diff_norms_after = try_np_load(log_dir / "after" / "normalized_grad_diff_l2_norms.npy")
    norms_after = try_np_load(log_dir / "after" / "grad_l2_norms.npy")
    losses_dmlmc = try_np_load(log_dir / "losses_delayed_mlmc.npy")
    times_dmlmc = try_np_load(log_dir / "times_delayed_mlmc.npy")
    losses_mlmc = try_np_load(log_dir / "losses_mlmc.npy")
    times_mlmc = try_np_load(log_dir / "time_mlmc.npy")
    losses_base = try_np_load(log_dir / "losses_baseline.npy")
    times_base = try_np_load(log_dir / "times_baseline.npy")
    # breakpoint()


if __name__ == "__main__":
    fire.Fire(main)
