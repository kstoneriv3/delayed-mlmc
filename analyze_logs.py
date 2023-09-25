import os
from pathlib import Path
from typing import Optional

import fire
import numpy as np


def get_latest_timestamp() -> str:
    timestamps = [ts for ts in os.listdir("./logs") if ts.startswith("202") and len(ts) == 14]
    return max(timestamps)


def main(timestamp: Optional[str] = None) -> None:
    timestamp = str(timestamp or get_latest_timestamp())
    log_dir = Path("./logs") / timestamp
    losses = np.load(log_dir / "losses.npy")
    norms = np.load(log_dir / "grad_l2_norms.npy")
    diff_norms = np.load(log_dir / "normalized_grad_diff_l2_norms.npy")


if __name__ == "__main__":
    fire.Fire(main)
