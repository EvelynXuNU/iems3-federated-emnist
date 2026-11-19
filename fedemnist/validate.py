"""Falsification and limiting-case experiments, with quantitative acceptance rules."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .core import (Client, Config, aggregate, evaluate_clients, model, round_updates,
                   run, save_json, setup, statistics, weights)


def distance(left, right):
    return max(float((left[key] - right[key]).abs().max()) for key in left)


def checks(root, output, with_ray):
    setup()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    config = Config(clients=4, epochs=2, rounds=3)
    clients = [Client(root, i, config) for i in range(100)]
    initial = weights(model(config.seed))

    # An unequal-count oracle distinguishes weighted FedAvg from an unweighted mean.
    result = aggregate([{"count": 1, "weights": {"w": torch.tensor([1.0])}},
                        {"count": 3, "weights": {"w": torch.tensor([9.0])}}])
    value = float(result["w"].item())
    results["unequal_count_oracle"] = {"value": value, "expected": 7.0,
                                       "unweighted_value": 5.0, "pass": value == 7.0}

    # Independent full-batch gradient oracle checks the local optimizer and initialization.
    full_config = replace(config, clients=1, epochs=1, batch_size=10000)
    client = Client(root, 0, full_config)
    actual = client.train(initial, 1)["weights"]
    oracle = model()
    oracle.load_state_dict(initial)
    loss = nn.functional.cross_entropy(oracle(client.x), client.y)
    gradients = torch.autograd.grad(loss, oracle.parameters())
    with torch.no_grad():
        for parameter, gradient in zip(oracle.parameters(), gradients):
            parameter -= full_config.lr * gradient
    error = distance(actual, weights(oracle))
    results["single_client_gradient_oracle"] = {"max_parameter_error": error,
                                               "tolerance": 1e-6, "pass": error < 1e-6}

    # Ten rounds at lr=0 should preserve the global model, even under sampling/noise.
    zero_config = replace(config, lr=0.0, noise=0.5)
    zero_clients = [Client(root, i, zero_config) for i in range(100)]
    state = initial
    generator = np.random.default_rng(123)
    trace = []
    for round_id in range(1, 11):
        selected = generator.choice(100, 4, replace=False)
        state = aggregate(round_updates(root, zero_clients, zero_config, selected, state, round_id))
        trace.append({"round": round_id, "max_parameter_error": distance(state, initial)})
    max_error = max(row["max_parameter_error"] for row in trace)
    results["zero_learning_rate"] = {"trace": trace, "max_parameter_error": max_error,
                                      "tolerance": 1e-6, "pass": max_error < 1e-6}

    # The b=0 branch must exactly preserve input tensors and identical client updates.
    duplicate = Client(root, 0, config)
    clean_error = float((duplicate.x - duplicate.clean).abs().max())
    update_error = distance(clients[0].train(initial, 1)["weights"],
                            duplicate.train(initial, 1)["weights"])
    results["zero_noise_identity"] = {"max_input_error": clean_error,
                                       "max_update_error": update_error,
                                       "pass": clean_error == 0 and update_error == 0}

    # The target noise law has mean zero and variance 2*b*b, with independent draws.
    noise_client = Client(root, 0, replace(config, noise=0.2))
    noise = (noise_client.x - noise_client.clean).numpy().ravel()
    results["noise_distribution"] = {
        "n": len(noise), "mean": float(noise.mean()), "variance": float(noise.var()),
        "expected_variance": 0.08, "adjacent_correlation": float(np.corrcoef(noise[:-1], noise[1:])[0, 1]),
        "mean_tolerance": 0.005, "variance_tolerance": 0.005,
        "pass": bool(abs(noise.mean()) < 0.005 and abs(noise.var() - 0.08) < 0.005)}

    # Can the architecture and optimizer memorize a tiny balanced set?
    xs, ys = [], []
    local_labels, local_counts = torch.unique(clients[0].y, return_counts=True)
    selected_labels = local_labels[local_counts >= 5][:10]
    if len(selected_labels) != 10:
        raise ValueError("The local capacity experiment requires ten classes with five examples in client 0.")
    for label in selected_labels:
        candidates = clients[0].clean[clients[0].y == label][:5]
        xs.append(candidates)
        ys.append(torch.full((len(candidates),), label, dtype=torch.long))
    x, y = torch.cat(xs), torch.cat(ys)
    tiny = model(17)
    optimizer = torch.optim.SGD(tiny.parameters(), lr=0.3)
    tiny_trace = []
    for step in range(401):
        if step % 20 == 0:
            metric = statistics(tiny, x, y)
            tiny_trace.append({"step": step, "accuracy": metric["correct"] / len(y),
                               "loss": metric["loss_sum"] / len(y)})
        if step < 400:
            optimizer.zero_grad(set_to_none=True)
            nn.functional.cross_entropy(tiny(x), y).backward()
            optimizer.step()
    results["tiny_set_memorization"] = {"n": len(y), "client": 0,
                                         "labels": selected_labels.tolist(), "trace": tiny_trace,
                                         "threshold": 0.98, "pass": tiny_trace[-1]["accuracy"] >= 0.98}

    # Non-IID stress: compare 10 rounds drawn only from the ten smallest/largest clients.
    sizes = np.array([len(client.y) for client in clients])
    stress = {}
    for name, pool in (("smallest", np.argsort(sizes)[:10]), ("largest", np.argsort(sizes)[-10:])):
        state = initial
        stress_trace = []
        for round_id in range(11):
            metrics = evaluate_clients(clients, state)
            stress_trace.append({"round": round_id, "val_acc": metrics["val_acc"],
                                 "client_p10": float(np.quantile(metrics["client_val_acc"], 0.1))})
            if round_id < 10:
                state = aggregate(round_updates(root, clients, config, pool, state, round_id + 1))
        stress[name] = {"clients": pool.tolist(), "total_training_samples": int(sizes[pool].sum()),
                         "trace": stress_trace}
    results["restricted_participation_stress"] = stress

    if with_ray:
        import ray
        try:
            serial_state, parallel_state = initial, initial
            generator = np.random.default_rng(999)
            ray_trace = []
            for round_id in range(1, 4):
                selected = generator.choice(100, 4, replace=False)
                serial = round_updates(root, clients, config, selected, serial_state, round_id)
                parallel = round_updates(root, clients, replace(config, backend="ray"),
                                         selected, parallel_state, round_id)
                serial_state, parallel_state = aggregate(serial), aggregate(parallel)
                events = sorted([(u["start"], 1) for u in parallel] + [(u["end"], -1) for u in parallel])
                active = peak = 0
                for _, change in events:
                    active += change
                    peak = max(peak, active)
                ray_trace.append({"round": round_id, "selected": selected.tolist(),
                                  "max_parameter_error": distance(serial_state, parallel_state),
                                  "distinct_workers": len({u["pid"] for u in parallel}),
                                  "peak_overlapping_updates": peak})
            results["ray_serial_equivalence"] = {"trace": ray_trace, "tolerance": 1e-6,
                "pass": all(row["max_parameter_error"] < 1e-6 and row["distinct_workers"] == 4
                            for row in ray_trace)}
        finally:
            ray.shutdown()
    save_json(output / "validation.json", results)
    failed = [name for name, result in results.items() if result.get("pass") is False]
    print(json.dumps(results, indent=2), flush=True)
    if failed:
        raise RuntimeError(f"Validation failed: {failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/clients")
    parser.add_argument("--output", default="results")
    parser.add_argument("--ray", action="store_true")
    parser.add_argument("--shuffled", action="store_true")
    parser.add_argument("--rounds", type=int, default=100)
    args = parser.parse_args()
    if args.shuffled:
        run(args.root, args.output, Config(shuffled=True, rounds=args.rounds))
    else:
        checks(args.root, args.output, args.ray)
