"""Generate report figures directly from the saved numerical experiment records."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from .core import Config, save_json
from .experiments import SCALES

COLORS = ["#156b82", "#d26b33", "#7057a3", "#528045", "#ba4566", "#555b68"]


def read(path):
    return json.loads(Path(path).read_text())


def histories(root, configs):
    return [read(Path(root) / config.name / "history.json") for config in configs]


def curve(ax, runs, metric, label, color):
    values = np.array([[row[metric] for row in run] for run in runs])
    rounds = np.arange(values.shape[1])
    mean, deviation = values.mean(axis=0), values.std(axis=0, ddof=1)
    ax.plot(rounds, mean, label=label, color=color, linewidth=1.7)
    ax.fill_between(rounds, mean - deviation, mean + deviation, color=color, alpha=0.12, linewidth=0)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Accuracy" if metric.endswith("acc") else "Cross-entropy loss")
    if metric.endswith("acc"):
        ax.yaxis.set_major_formatter(PercentFormatter(1))


def generate(root="results", destination="output/figures"):
    root, destination = Path(root), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.16,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})

    def finish(fig, name):
        fig.savefig(destination / f"{name}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    selection = read(root / "selection.json")
    seeds, rounds = selection["seeds"], selection["rounds"]
    epochs = selection["selected"]["epochs"]
    manifest = read("data/clients/manifest.json")
    clients = manifest["clients"]
    counts = np.array([row["class_counts"] for row in clients])
    sizes = counts.sum(axis=1)
    selected_clients = np.argsort(sizes)[[0, 25, 50, 75, 99]].tolist()
    fig, axes = plt.subplots(3, 2, figsize=(9.2, 7.5), layout="constrained")
    for ax, client_id in zip(axes.flat, [None] + selected_clients):
        values = counts.sum(axis=0) if client_id is None else counts[client_id]
        ax.bar(np.arange(62), values, width=0.85, color=COLORS[0])
        title = "All 100 clients" if client_id is None else f"Client {client_id}"
        ax.set_title(f"{title} | n={values.sum():,} | {np.count_nonzero(values)} classes", loc="left")
        ax.set_xlabel("Class ID")
        ax.set_ylabel("Examples")
        ax.set_xticks([0, 10, 20, 30, 40, 50, 61])
    finish(fig, "distribution")

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4), layout="constrained")
    for index, row in enumerate(selection["grid"]):
        configs = [Config(clients=row["clients"], epochs=row["epochs"], seed=s, rounds=rounds) for s in seeds]
        runs = histories(root, configs)
        label = f"C={row['clients']}%, E={row['epochs']}"
        for ax, metric in zip(axes.flat, ("train_loss", "val_loss", "train_acc", "val_acc")):
            curve(ax, runs, metric, label, COLORS[index])
    for ax, title in zip(axes.flat, ("Training loss", "Validation loss", "Training accuracy", "Validation accuracy")):
        ax.set_title(title, loc="left")
    axes[0, 0].legend(fontsize=8)
    finish(fig, "hyperparameters")

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4), layout="constrained")
    for index, backend in enumerate(("serial", "ray")):
        runs = histories(root, [Config(clients=4, epochs=epochs, seed=s, rounds=rounds,
                                       backend=backend) for s in seeds])
        for ax, metric in zip(axes.flat, ("train_loss", "val_loss", "train_acc", "val_acc")):
            curve(ax, runs, metric, f"{backend}, 4 clients", COLORS[index])
    for ax, title in zip(axes.flat, ("Training loss", "Validation loss", "Training accuracy", "Validation accuracy")):
        ax.set_title(title, loc="left")
    axes[0, 0].legend()
    finish(fig, "ray_curves")

    ray_history = histories(root, [Config(clients=4, epochs=epochs, seed=seeds[0], rounds=rounds,
                                         backend="ray")])[0]
    fig, ax = plt.subplots(figsize=(9.2, 2.8), layout="constrained")
    offset = 0
    timeline = []
    for row in ray_history[1:4]:
        first = min(interval["start"] for interval in row["intervals"])
        last = max(interval["end"] for interval in row["intervals"])
        for index, interval in enumerate(row["intervals"]):
            start = offset + interval["start"] - first
            duration = interval["end"] - interval["start"]
            ax.broken_barh([(start, duration)], (index - 0.32, 0.64), facecolors=COLORS[row["round"] - 1])
            timeline.append({"round": row["round"], **interval})
        offset += last - first + 0.04
    ax.set_yticks(range(4), ["Selected actor 1", "Selected actor 2", "Selected actor 3", "Selected actor 4"])
    ax.set_xlabel("Seconds within the concatenated update windows (rounds 1-3)")
    ax.set_title("Actor execution overlap; startup gaps omitted", loc="left")
    finish(fig, "ray_timeline")

    picked = selection["selected"]
    baseline_config = Config(clients=picked["clients"], epochs=picked["epochs"], seed=seeds[0], rounds=rounds)
    baseline = histories(root, [baseline_config])[0]
    client_accuracy = np.array([row["client_val_acc"] for row in baseline]).T
    client_order = np.argsort(client_accuracy[:, -1])
    participation = np.bincount([i for row in baseline for i in row["selected"]], minlength=100)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4), layout="constrained", width_ratios=[1.4, 1])
    im = axes[0].imshow(client_accuracy[client_order], aspect="auto", origin="lower", vmin=0, vmax=1, cmap="viridis")
    axes[0].grid(False)
    axes[0].set_xlabel("Communication round")
    axes[0].set_ylabel("Clients ranked by final validation accuracy")
    axes[0].set_title("Which clients benefit as training proceeds?", loc="left")
    fig.colorbar(im, ax=axes[0], label="Validation accuracy", shrink=0.8)
    axes[1].scatter(participation, client_accuracy[:, -1], s=sizes / 9, alpha=0.65, color=COLORS[0])
    axes[1].set_xlabel("Rounds selected")
    axes[1].set_ylabel("Final client validation accuracy")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set_title("Circle area follows client sample count", loc="left", fontsize=9)
    finish(fig, "client_outcomes")

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), layout="constrained")
    for index, scale in enumerate(SCALES):
        configs = [Config(epochs=selection["noise_epochs"], noise=scale, seed=s, rounds=rounds) for s in seeds]
        runs = histories(root, configs)
        curve(axes[0], runs, "train_acc", f"b={scale:g}", COLORS[index])
        curve(axes[1], runs, "val_acc", f"b={scale:g}", COLORS[index])
    axes[0].set_title("Training on each fixed perturbed dataset", loc="left")
    axes[1].set_title("Validation on clean images", loc="left")
    axes[1].legend(fontsize=8)
    finish(fig, "noise_curves")

    tests = read(root / "test_results.json")
    noise_table, recall = [], []
    for scale in SCALES:
        configs = [Config(epochs=selection["noise_epochs"], noise=scale, seed=s, rounds=rounds) for s in seeds]
        summaries = [read(root / c.name / "summary.json") for c in configs]
        row = {"noise": scale}
        for metric in ("train_acc", "clean_train_acc", "val_acc"):
            values = [summary[metric] for summary in summaries]
            row[metric] = float(np.mean(values))
            row[metric + "_sd"] = float(np.std(values, ddof=1))
        values = [tests[c.name]["accuracy"] for c in configs]
        row["test_acc"], row["test_acc_sd"] = float(np.mean(values)), float(np.std(values, ddof=1))
        confusion = sum(np.array(tests[c.name]["confusion"]) for c in configs)
        recall.append(np.divide(confusion.diagonal(), confusion.sum(axis=1),
                                out=np.zeros(62), where=confusion.sum(axis=1) > 0))
        noise_table.append(row)
    fig, ax = plt.subplots(figsize=(8.6, 4.3), layout="constrained")
    for index, (metric, label) in enumerate((("train_acc", "Perturbed training"),
                                            ("clean_train_acc", "Clean training"), ("test_acc", "Clean test"))):
        ax.errorbar(SCALES, [r[metric] for r in noise_table], yerr=[r[metric + "_sd"] for r in noise_table],
                    label=label, color=COLORS[index], marker="o", capsize=3)
    ax.set_xlabel("Laplace noise scale b (pixel range before noise is 0-1)")
    ax.set_ylabel("Final accuracy")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend()
    finish(fig, "noise_tradeoff")

    recall = np.array(recall).T
    class_order = np.argsort(counts.sum(axis=0))
    differences = 100 * (recall - recall[:, :1])
    fig, ax = plt.subplots(figsize=(8.7, 7.5), layout="constrained")
    im = ax.imshow(differences[class_order], aspect="auto", cmap="RdBu", vmin=-50, vmax=50)
    ax.grid(False)
    ax.set_xticks(range(len(SCALES)), [str(s) for s in SCALES])
    ticks = np.arange(0, 62, 3)
    ax.set_yticks(ticks, [f"{k} (n={counts[:, k].sum()})" for k in class_order[ticks]])
    ax.set_xlabel("Noise scale b")
    ax.set_ylabel("Class ID (overall sample count), rarest at top")
    ax.set_title("Class recall changes relative to b=0 on clean test images", loc="left")
    fig.colorbar(im, ax=ax, label="Change in recall (percentage points)", shrink=0.7)
    finish(fig, "class_noise_sensitivity")

    validation = read(root / "validation.json")
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.0), layout="constrained")
    tiny = validation["tiny_set_memorization"]["trace"]
    axes[0].plot([r["step"] for r in tiny], [r["accuracy"] for r in tiny], color=COLORS[0])
    axes[0].set_xlabel("Gradient step")
    axes[0].set_ylabel("50-example training accuracy")
    axes[0].set_title("Capacity check", loc="left")
    for index, (name, group) in enumerate(validation["restricted_participation_stress"].items()):
        axes[1].plot([r["round"] for r in group["trace"]], [r["val_acc"] for r in group["trace"]],
                     label=f"10 {name} clients", color=COLORS[index])
    axes[1].legend(fontsize=7)
    axes[1].set_xlabel("Communication round")
    axes[1].set_ylabel("Global validation accuracy")
    axes[1].set_title("Restricted participation", loc="left")
    shuffled = read(root / Config(shuffled=True, rounds=rounds).name / "history.json")
    matched = read(root / Config(rounds=rounds).name / "history.json")
    for index, (label, rows) in enumerate((("True labels", matched), ("Shuffled local labels", shuffled))):
        axes[2].plot([r["round"] for r in rows], [r["val_acc"] for r in rows], label=label, color=COLORS[index])
    axes[2].legend(fontsize=7)
    axes[2].set_xlabel("Communication round")
    axes[2].set_ylabel("Clean validation accuracy")
    axes[2].set_title("Label-signal falsification", loc="left")
    for ax in axes:
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    finish(fig, "validation_experiments")

    # Numerical companion to every aggregate in the report.
    save_json(root / "analysis.json", {
        "selected_clients": selected_clients, "noise_table": noise_table,
        "client_seed": seeds[0], "client_val_p10": float(np.quantile(client_accuracy[:, -1], .1)),
        "client_val_median": float(np.median(client_accuracy[:, -1])),
        "client_val_min": float(client_accuracy[:, -1].min()),
        "client_val_max": float(client_accuracy[:, -1].max()),
        "participation_min": int(participation.min()), "participation_max": int(participation.max()),
        "participation_accuracy_correlation": float(np.corrcoef(participation, client_accuracy[:, -1])[0, 1]),
        "shuffled_validation_acc": shuffled[-1]["val_acc"],
        "class_recall_by_noise": recall.tolist(), "ray_timeline": timeline})


if __name__ == "__main__":
    generate()
