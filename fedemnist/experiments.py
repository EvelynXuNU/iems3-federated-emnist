"""Run prespecified training comparisons, select on validation, then evaluate test."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .core import Config, model, run, save_json, setup, statistics

SEEDS = (17, 29, 43)
SCALES = (0.0, 0.05, 0.1, 0.2, 0.5, 1.0)


def suite(root, output, rounds):
    grid = []
    for count in (4, 10):
        for epochs in (1, 3):
            group = [run(root, output, Config(clients=count, epochs=epochs, seed=s, rounds=rounds))
                     for s in SEEDS]
            grid.append({"clients": count, "epochs": epochs,
                         "mean_val_acc": float(np.mean([r["val_acc"] for r in group]))})
    selected = max(grid, key=lambda row: row["mean_val_acc"])
    # Part 2 explicitly requires C=10%. Select its E using only C=10% validation results.
    noise_base = max((row for row in grid if row["clients"] == 10), key=lambda row: row["mean_val_acc"])
    noise_results = []
    for scale in SCALES:
        group = [run(root, output, Config(epochs=noise_base["epochs"], seed=s, noise=scale, rounds=rounds))
                 for s in SEEDS]
        noise_results.append({"noise": scale, "mean_val_acc": float(np.mean([r["val_acc"] for r in group]))})
    baseline_accuracy = noise_results[0]["mean_val_acc"]
    # Prespecified utility tolerance: at most two percentage points below clean validation.
    eligible = [row for row in noise_results if row["noise"] > 0
                and row["mean_val_acc"] >= baseline_accuracy - 0.02]
    selected_noise = max(eligible, key=lambda row: row["noise"])["noise"] if eligible else 0.0
    selection = {"seeds": list(SEEDS), "rounds": rounds, "grid": grid,
                 "selected": selected, "noise_epochs": noise_base["epochs"],
                 "noise_results": noise_results, "selected_noise": selected_noise,
                 "noise_selection_rule": "largest positive b within 0.02 of mean clean validation accuracy",
                 "test_used_for_selection": False}
    save_json(Path(output) / "selection.json", selection)
    for s in SEEDS:
        run(root, output, Config(clients=4, epochs=selected["epochs"], seed=s,
                                 rounds=rounds, backend="ray"))
    return selection


def evaluate(output, source):
    setup()
    output = Path(output)
    selection = json.loads((output / "selection.json").read_text())
    picked = selection["selected"]
    configs = []
    for seed in selection["seeds"]:
        configs.append(Config(clients=picked["clients"], epochs=picked["epochs"], seed=seed,
                              rounds=selection["rounds"]))
        configs.append(Config(clients=4, epochs=picked["epochs"], seed=seed,
                              rounds=selection["rounds"], backend="ray"))
        configs.append(Config(clients=4, epochs=picked["epochs"], seed=seed,
                              rounds=selection["rounds"]))
        configs.extend(Config(epochs=selection["noise_epochs"], seed=seed, noise=scale,
                              rounds=selection["rounds"]) for scale in SCALES)
    # This is the only experiment-stage entry point that opens the held-out test file.
    data = np.load(Path(source) / "test_data.npy", allow_pickle=True)[0]
    x = torch.from_numpy(1.0 - np.asarray(data["images"], dtype=np.float32))
    y = torch.from_numpy(np.asarray(data["labels"], dtype=np.int64))
    results = {}
    for config in configs:
        if config.name in results:
            continue
        network = model()
        network.load_state_dict(torch.load(output / config.name / "model.pt", weights_only=True))
        metrics = statistics(network, x, y)
        results[config.name] = {"accuracy": metrics["correct"] / metrics["count"],
                                "loss": metrics["loss_sum"] / metrics["count"], "n": len(y),
                                "confusion": metrics["confusion"].tolist()}
        print(config.name, f"test={results[config.name]['accuracy']:.4f}", flush=True)
    save_json(output / "test_results.json", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("pilot", "suite", "test"))
    parser.add_argument("--root", default="data/clients")
    parser.add_argument("--output", default="results")
    parser.add_argument("--source", default="Assignment3-data")
    parser.add_argument("--rounds", type=int, default=100)
    args = parser.parse_args()
    if args.stage == "pilot":
        run(args.root, "tmp/pilot", Config(rounds=args.rounds))
    elif args.stage == "suite":
        try:
            suite(args.root, args.output, args.rounds)
        finally:
            import ray
            ray.shutdown()
    else:
        evaluate(args.output, args.source)
