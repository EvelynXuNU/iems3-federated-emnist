"""Create deterministic client-local files from the course-supplied training data."""

import argparse
import hashlib
from pathlib import Path

import numpy as np

from .core import save_json


def prepare(source, destination):
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    path = source / "train_data.npy"
    # The course file is a trusted local object array and requires pickle loading.
    data = np.load(path, allow_pickle=True)
    if len(data) != 100:
        raise ValueError("Expected the assignment's 100 clients.")
    records = []
    for client_id, client in enumerate(data):
        x = np.asarray(client["images"], dtype=np.float32)
        y = np.asarray(client["labels"], dtype=np.int64)
        if x.shape != (len(y), 28, 28) or len(y) < 2:
            raise ValueError(f"Client {client_id} needs at least two valid 28x28 images.")
        if not np.isfinite(x).all() or x.min() < 0 or x.max() > 1 or y.min() < 0 or y.max() >= 62:
            raise ValueError("Expected finite 0-1 pixels and integer labels from 0 to 61.")
        x = 1.0 - x
        order = np.random.default_rng([2026, client_id]).permutation(len(y))
        boundary = int(0.8 * len(y))
        train, val = order[:boundary], order[boundary:]
        np.savez(destination / f"client_{client_id:03d}.npz", train_x=x[train], train_y=y[train],
                 val_x=x[val], val_y=y[val], train_indices=train, val_indices=val)
        records.append({"client": client_id, "n": len(y), "train_n": len(train), "val_n": len(val),
                        "class_counts": np.bincount(y, minlength=62).tolist(),
                        "train_class_counts": np.bincount(y[train], minlength=62).tolist(),
                        "val_class_counts": np.bincount(y[val], minlength=62).tolist()})
    save_json(destination / "manifest.json", {
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "preprocessing": "float32, invert grayscale with 1-x; no further scaling",
        "split_seed": 2026, "split_rule": "floor(0.8*n) random training indices per client",
        "clients": records})
    print(f"Prepared {len(records)} clients: {sum(r['train_n'] for r in records)} train, "
          f"{sum(r['val_n'] for r in records)} validation examples.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="Assignment3-data")
    parser.add_argument("--destination", default="data/clients")
    args = parser.parse_args()
    prepare(args.source, args.destination)
