"""Client-local learning and sample-weighted FedAvg, implemented directly."""

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import os
import platform
import time

import numpy as np
import ray
import torch
from torch import nn


def setup():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


@dataclass(frozen=True)
class Config:
    clients: int = 10
    epochs: int = 3
    lr: float = 0.1
    rounds: int = 100
    batch_size: int = 32
    seed: int = 17
    noise: float = 0.0
    backend: str = "serial"
    shuffled: bool = False

    def __post_init__(self):
        if not 1 <= self.clients <= 10:
            raise ValueError("Select between 1 and 10 of the 100 clients.")
        if self.epochs < 1 or self.rounds < 1 or self.batch_size < 1:
            raise ValueError("Epochs, rounds and batch size must be positive.")
        if self.noise < 0 or self.lr < 0 or self.backend not in ("serial", "ray"):
            raise ValueError("Invalid noise, learning rate or backend.")

    @property
    def name(self):
        suffix = "_shuffled" if self.shuffled else ""
        return (f"{self.backend}_c{self.clients}_e{self.epochs}"
                f"_b{self.noise:g}_s{self.seed}_r{self.rounds}{suffix}")


def model(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 62))


def weights(network):
    return {key: value.detach().clone() for key, value in network.state_dict().items()}


def aggregate(updates):
    """Weight each client's model by its number of local training examples."""
    counts = [update["count"] for update in updates]
    if not counts or min(counts) <= 0:
        raise ValueError("FedAvg requires nonempty updates with positive sample counts.")
    total = sum(counts)
    return {
        key: sum(update["weights"][key] * (update["count"] / total) for update in updates)
        for key in updates[0]["weights"]
    }


class Client:
    """Load one client file. Only weights, sample counts and diagnostics leave training."""

    def __init__(self, root, client_id, config):
        setup()
        self.client_id = client_id
        self.config = config
        with np.load(Path(root) / f"client_{client_id:03d}.npz") as data:
            self.clean = torch.from_numpy(data["train_x"].copy())
            self.y = torch.from_numpy(data["train_y"].copy())
            self.val_x = torch.from_numpy(data["val_x"].copy())
            self.val_y = torch.from_numpy(data["val_y"].copy())
        # One release per stored training image. Reuse that perturbation across rounds.
        generator = np.random.default_rng(np.random.SeedSequence([config.seed, client_id, 707]))
        self.x = self.clean.clone()
        if config.noise > 0:
            noise = generator.laplace(0, config.noise, size=self.x.shape).astype(np.float32)
            self.x += torch.from_numpy(noise)
        if config.shuffled:
            permutation = np.random.default_rng([config.seed, client_id, 808]).permutation(len(self.y))
            self.y = self.y[permutation]

    def train(self, state, round_id):
        start = time.time()
        network = model()
        network.load_state_dict(state)
        network.train()
        optimizer = torch.optim.SGD(network.parameters(), lr=self.config.lr)
        generator = torch.Generator().manual_seed(
            self.config.seed * 1_000_000 + round_id * 100 + self.client_id
        )
        for _ in range(self.config.epochs):
            order = torch.randperm(len(self.y), generator=generator)
            for batch in order.split(self.config.batch_size):
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(network(self.x[batch]), self.y[batch])
                loss.backward()
                optimizer.step()
        return {"weights": weights(network), "count": len(self.y),
                "client": self.client_id, "start": start, "end": time.time(), "pid": os.getpid()}

    def evaluate(self, network):
        # Return sufficient statistics, including the class confusion matrix.
        training = statistics(network, self.x, self.y)
        clean_training = statistics(network, self.clean, self.y) if self.config.noise > 0 else training
        return {"train": training,
                "clean_train": clean_training,
                "val": statistics(network, self.val_x, self.val_y)}


RayClient = ray.remote(num_cpus=1, num_gpus=0, max_restarts=0)(Client)


@torch.no_grad()
def statistics(network, x, y):
    network.eval()
    total_loss = 0.0
    confusion = np.zeros((62, 62), dtype=np.int64)
    for start in range(0, len(y), 512):
        target = y[start:start + 512]
        logits = network(x[start:start + 512])
        total_loss += nn.functional.cross_entropy(logits, target, reduction="sum").item()
        prediction = logits.argmax(dim=1)
        confusion += np.bincount((target * 62 + prediction).numpy(), minlength=62 * 62).reshape(62, 62)
    return {"count": len(y), "loss_sum": total_loss, "correct": int(confusion.trace()),
            "confusion": confusion}


def evaluate_clients(clients, state):
    network = model()
    network.load_state_dict(state)
    local = [client.evaluate(network) for client in clients]
    output = {}
    for split in ("train", "clean_train", "val"):
        count = sum(row[split]["count"] for row in local)
        output[f"{split}_loss"] = sum(row[split]["loss_sum"] for row in local) / count
        output[f"{split}_acc"] = sum(row[split]["correct"] for row in local) / count
    output["client_val_acc"] = [row["val"]["correct"] / row["val"]["count"] for row in local]
    output["val_confusion"] = sum(row["val"]["confusion"] for row in local)
    return output


def start_ray():
    import ray
    if not ray.is_initialized():
        ray.init(num_cpus=4, num_gpus=0, include_dashboard=False, log_to_driver=False)


def round_updates(root, clients, config, selected, state, round_id):
    if config.backend == "serial":
        return [clients[int(client_id)].train(state, round_id) for client_id in selected]
    start_ray()
    actors = []
    try:
        # Only the selected clients become Actors. Each reads only its own file.
        actors = [RayClient.remote(str(Path(root).resolve()), int(i), config) for i in selected]
        state_reference = ray.put(state)
        return ray.get([actor.train.remote(state_reference, round_id) for actor in actors], timeout=180)
    finally:
        for actor in actors:
            ray.kill(actor, no_restart=True)


def run(root, destination, config, force=False):
    setup()
    destination = Path(destination) / config.name
    manifest = json.loads((Path(root) / "manifest.json").read_text())
    if (destination / "summary.json").exists() and not force:
        existing = json.loads((destination / "summary.json").read_text())
        if existing["config"] != asdict(config) or existing["data_sha256"] != manifest["source_sha256"]:
            raise ValueError("Existing result does not match the configuration or data.")
        return existing
    destination.mkdir(parents=True, exist_ok=True)
    clients = [Client(root, i, config) for i in range(100)]
    state = weights(model(config.seed))
    generator = np.random.default_rng([config.seed, 909])
    started = time.perf_counter()
    history = []
    participation = np.zeros(100, dtype=int)
    final_confusion = None
    for round_id in range(config.rounds + 1):
        selected = []
        intervals = []
        round_started = time.perf_counter()
        if round_id:
            selected = generator.choice(100, config.clients, replace=False).tolist()
            updates = round_updates(root, clients, config, selected, state, round_id)
            state = aggregate(updates)
            participation[selected] += 1
            intervals = [{key: update[key] for key in ("client", "start", "end", "pid")}
                         for update in updates]
        update_seconds = time.perf_counter() - round_started
        metrics = evaluate_clients(clients, state)
        final_confusion = metrics.pop("val_confusion")
        row = {"round": round_id, "selected": selected, "update_seconds": update_seconds,
               "intervals": intervals, **metrics}
        history.append(row)
        save_json(destination / "history.json", history)
        if round_id % 20 == 0 or round_id == config.rounds:
            print(f"{config.name} round {round_id}: train={metrics['train_acc']:.3f} "
                  f"val={metrics['val_acc']:.3f}", flush=True)
    torch.save(state, destination / "model.pt")
    np.save(destination / "confusion.npy", final_confusion)
    summary = {"config": asdict(config), "data_sha256": manifest["source_sha256"],
               "environment": {"python": platform.python_version(), "system": platform.system(),
                               "machine": platform.machine(), "torch": torch.__version__,
                               "ray": ray.__version__, "numpy": np.__version__,
                               "device": "cpu", "torch_threads": 1},
               "seconds": time.perf_counter() - started,
               "participation": participation.tolist(), **history[-1]}
    save_json(destination / "summary.json", summary)
    return summary
