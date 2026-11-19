"""Build the PDF report from saved results and verified figures."""

from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                               Spacer, Table, TableStyle)

from .core import Config
from .plots import read

REPO = "https://github.com/EvelynXuNU/iems3-federated-emnist"


def build():
    root = Path("results")
    selection = read(root / "selection.json")
    analysis = read(root / "analysis.json")
    validation = read(root / "validation.json")
    tests = read(root / "test_results.json")
    manifest = read("data/clients/manifest.json")
    clients = manifest["clients"]
    counts = np.array([row["class_counts"] for row in clients])
    train_counts = np.array([row["train_class_counts"] for row in clients])
    val_counts = np.array([row["val_class_counts"] for row in clients])
    sizes = counts.sum(axis=1)
    seeds, rounds = selection["seeds"], selection["rounds"]
    chosen = selection["selected"]
    noise_epochs = selection["noise_epochs"]
    b = selection["selected_noise"]
    noise_rows = analysis["noise_table"]
    clean_noise_row = noise_rows[0]
    chosen_noise_row = next(row for row in noise_rows if row["noise"] == b)
    baseline_configs = [Config(clients=chosen["clients"], epochs=chosen["epochs"], seed=s, rounds=rounds) for s in seeds]
    ray_configs = [Config(clients=4, epochs=chosen["epochs"], seed=s, rounds=rounds, backend="ray") for s in seeds]

    def summaries(configs):
        return [read(root / config.name / "summary.json") for config in configs]

    def average(configs, field):
        values = [row[field] for row in summaries(configs)]
        return float(np.mean(values)), float(np.std(values, ddof=1))

    def accuracy(configs):
        values = [tests[config.name]["accuracy"] for config in configs]
        return float(np.mean(values)), float(np.std(values, ddof=1))

    def pct(value):
        return f"{100 * value:.2f}%"

    def pair(values):
        return f"{100 * values[0]:.2f} +/- {100 * values[1]:.2f}"

    teal = colors.HexColor("#14576a")
    muted = colors.HexColor("#52616b")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("BodyCustom", fontName="Helvetica", fontSize=10, leading=14,
                              spaceAfter=8, textColor=colors.HexColor("#25343c")))
    styles.add(ParagraphStyle("TitleCustom", fontName="Helvetica-Bold", fontSize=25, leading=29,
                              textColor=teal, spaceAfter=14))
    styles.add(ParagraphStyle("SectionCustom", fontName="Helvetica-Bold", fontSize=17, leading=21,
                              textColor=teal, spaceAfter=12))
    styles.add(ParagraphStyle("SmallCustom", fontName="Helvetica", fontSize=8.3, leading=11,
                              textColor=muted, spaceAfter=7))
    styles.add(ParagraphStyle("CellCustom", fontName="Helvetica", fontSize=8.3, leading=11))
    styles.add(ParagraphStyle("CodeCustom", fontName="Courier", fontSize=8.1, leading=12,
                              spaceAfter=8, textColor=colors.HexColor("#25343c")))
    story = []

    def p(text, style="BodyCustom"):
        story.append(Paragraph(text, styles[style]))

    def page(title):
        if story:
            story.append(PageBreak())
        p(title, "SectionCustom")

    def table(rows, widths):
        cells = [[Paragraph(escape(str(cell)), styles["CellCustom"]) for cell in row] for row in rows]
        tab = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
        tab.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e4eff2")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LINEBELOW", (0, 0), (-1, 0), 0.8, teal),
            ("LINEBELOW", (0, 1), (-1, -1), 0.35, colors.HexColor("#dbe2e5")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(tab)
        story.append(Spacer(1, 9))

    def figure(name, caption, width=505):
        path = Path("output/figures") / f"{name}.png"
        with PILImage.open(path) as im:
            height = width * im.height / im.width
        story.append(Image(str(path), width=width, height=height))
        story.append(Spacer(1, 5))
        p(caption, "SmallCustom")

    p("Federated learning and<br/>Laplace input perturbation", "TitleCustom")
    p("Assignment 3 | Federated EMNIST | November 2026", "SmallCustom")
    p("GitHub username <b>EvelynXuNU</b><br/>Repository <b>iems3-federated-emnist</b><br/>"
      f'<link href="{REPO}" color="#14576a">{REPO}</link>')
    p(f"A sample-weighted FedAvg implementation trains a 784-128-62 network across 100 simulated clients. "
      f"The selected clean configuration uses C={chosen['clients']}% and E={chosen['epochs']}, "
      f"with mean clean test accuracy {pct(accuracy(baseline_configs)[0])}. "
      f"The validation-based noise choice is b={b:g}, with clean test accuracy {pct(chosen_noise_row['test_acc'])}.")
    table([["Final model", "Validation accuracy (%)", "Clean test accuracy (%)"],
           [f"Serial C={chosen['clients']}%, E={chosen['epochs']}", pair(average(baseline_configs, "val_acc")), pair(accuracy(baseline_configs))],
           [f"Ray C=4%, E={chosen['epochs']}", pair(average(ray_configs, "val_acc")), pair(accuracy(ray_configs))],
           [f"Laplace C=10%, E={noise_epochs}, b={b:g}",
            pair((chosen_noise_row['val_acc'], chosen_noise_row['val_acc_sd'])),
            pair((chosen_noise_row['test_acc'], chosen_noise_row['test_acc_sd']))]], [230, 140, 140])
    p("All accuracy summaries report mean +/- sample standard deviation over seeds 17, 29, and 43. "
      "They quantify training randomness on one fixed data split. Test images are clean for every model.")
    p("The validation evidence includes a weighted aggregation oracle, an independent gradient oracle, "
      "zero-update and zero-noise limits, tiny-set memorization, a shuffled-label control, restricted "
      "participation stress, and matched serial/Ray execution. These experiments target mathematical "
      "behavior, learnability and sensitivity to data conditions.")
    p("The implementation uses standard PyTorch and NumPy plus Ray Core. Each selected actor receives "
      "global weights and reads its own client file. Model updates and sample counts implement the server interface. "
      "The single-machine evaluator receives aggregated metrics from the simulated client objects.")
    p("The utility experiments use publicly reproducible pseudorandom seeds. Formal deployment privacy "
      "would require secret randomness, a declared adjacency relation, label protection, and accounting for "
      "all released statistics. The noise-choice discussion derives the image-sensitivity implication.")

    page("Validation evidence by assignment part")
    p("This guide connects each assignment part to its edge cases, falsifiable predictions, measured "
      "evidence and non-standard diagnostic. The following sections explain the experiments and their limits.")
    ray_trace = validation["ray_serial_equivalence"]["trace"]
    law = validation["noise_distribution"]
    majority = int(train_counts.sum(axis=0).argmax())
    majority_acc = val_counts[:, majority].sum() / val_counts.sum()
    table([["Assignment part", "Edge case and handling", "Prediction and measured evidence", "Non-standard figure"],
           ["Part 1.1: FedAvg",
            "Unequal client sizes use sample-count weights. Locally missing classes retain all 62 output units. Client-level metrics expose uneven outcomes.",
            f"Weighted oracle: expect 7, observe {validation['unequal_count_oracle']['value']:g} (an unweighted mean gives 5). "
            f"Single-client SGD: error {validation['single_client_gradient_oracle']['max_parameter_error']:.2e} < 1e-6. "
            f"Tiny-set capacity: {pct(validation['tiny_set_memorization']['trace'][-1]['accuracy'])} >= 98%. "
            f"Destroying image-label association reduces validation to {pct(analysis['shuffled_validation_acc'])}; majority baseline {pct(majority_acc)}.",
            "Figure 3: writer trajectories, final accuracy and selection frequency. Figure 9: capacity and restricted-participation stress."],
           ["Part 1.2: Ray",
            "Create only four selected actors per round, each with one CPU and zero GPUs. Cleanup runs even after failure; incomplete rounds abort. Crash injection is deferred.",
            "Matched clients and randomness should reproduce serial FedAvg. "
            f"Across three rounds, maximum weight error is {max(row['max_parameter_error'] for row in ray_trace):.2e} < 1e-6; "
            f"each round uses four distinct workers. Peak simultaneous updates range from {min(row['peak_overlapping_updates'] for row in ray_trace)} "
            f"to {max(row['peak_overlapping_updates'] for row in ray_trace)}, confirming overlap.",
            "Figure 5: actual actor update intervals in rounds 1-3. Overlap checks execution concurrency alongside weight equivalence."],
           ["Part 2: Laplace noise",
            "At b=0 preserve inputs exactly. Keep perturbed pixels unbounded as specified; labels and evaluation inputs stay clean. Diagnose class-level degradation at large b.",
            "Zero-noise input and update errors are exactly zero. "
            f"At b=0.2, expect mean 0 and variance 0.08; observe {law['mean']:.6f} and {law['variance']:.6f} "
            "(both within 0.005). "
            f"At b=1, perturbed training accuracy is {pct(noise_rows[-1]['train_acc'])} but clean test accuracy is "
            f"{pct(noise_rows[-1]['test_acc'])}, exposing poor transfer.",
            "Figure 8: changes in per-class test recall, ordered by class frequency. Figure 7 gives the required final-accuracy tradeoff."]],
          [58, 124, 220, 108])
    p("<b>Reproduce the validation evidence</b> after data preparation and dependency installation:")
    p("python -m fedemnist.validate --ray<br/>"
      "python -m fedemnist.validate --shuffled", "CodeCustom")
    p("The first command runs the aggregation and gradient oracles, zero-rate and zero-noise limits, noise-law "
      "check, tiny-set capacity, restricted-participation stress and serial/Ray comparison. The second runs the "
      "100-round shuffled-label control. Outputs are saved in results/validation.json and the shuffled run history. "
      "Threshold checks pass; participation stress and class diagnostics are descriptive experiments. The full "
      "reproduction sequence appears in the extended-validation section and README.", "SmallCustom")

    page("Data distribution and local splits")
    p(f"The course training archive contains {int(sizes.sum()):,} 28x28 images from 100 writers. Client sizes "
      f"range from {int(sizes.min())} to {int(sizes.max())}, with median {np.median(sizes):g}. "
      f"Each client is independently split into floor(0.8n) training examples and the remainder for validation, "
      f"giving {int(train_counts.sum()):,} training and {int(val_counts.sum()):,} validation examples.")
    figure("distribution", "Figure 1. Class counts in the complete local-data collection and five writers selected at client-size ranks 1, 26, 51, 76, and 100. Class IDs are retained to avoid assuming a label-to-glyph mapping.", width=490)
    p(f"Class {counts.sum(0).argmax()} has {counts.sum(0).max():,} examples, while class "
      f"{counts.sum(0).argmin()} has {counts.sum(0).min():,}. Across writers, the number of observed classes ranges "
      f"from {np.count_nonzero(counts, axis=1).min()} to {np.count_nonzero(counts, axis=1).max()}. "
      "Both sample-count imbalance and different label mixtures make local objectives heterogeneous.")
    p("Pixels are already in [0,1]. Preprocessing applies 1-x so the white background becomes zero. "
      "The identical transform is applied to training, validation, and test images before any training noise is added.")

    page("FedAvg training and hyperparameter comparison")
    p("Each selected client starts from the same global model and performs E epochs of shuffled minibatch SGD, "
      "using learning rate 0.1 and batch size 32. The hidden layer has 128 units and ReLU. The output has 62 logits "
      "and cross-entropy loss. The optimizer has no momentum and is reset for every local update.")
    p("For selected clients S, the server computes w(next) = sum[n(k) w(k)] / sum[n(k)], where n(k) is the "
      "local training count. Reported losses and accuracies are likewise weighted by the number of evaluated examples "
      "across all 100 clients, including clients absent from the current update round.")
    rows = [["C", "Local E", "Training accuracy (%)", "Validation accuracy (%)"]]
    for row in selection["grid"]:
        configs = [Config(clients=row["clients"], epochs=row["epochs"], seed=s, rounds=rounds) for s in seeds]
        rows.append([f"{row['clients']}%", row["epochs"], pair(average(configs, "train_acc")), pair(average(configs, "val_acc"))])
    table(rows, [55, 65, 195, 195])
    figure("hyperparameters", "Figure 2. Mean curves over three seeds; bands show one sample standard deviation. Final validation accuracy selects the configuration in the first-page table.", width=490)
    p("E=3 improves the final accuracy for both participation fractions. More local steps also mean more "
      "computation per round. Thus this is a comparison at a fixed 100-round communication budget. It does "
      "not isolate total optimization work. Selection uses the final round and averages the three validation results.")

    page("Client outcomes and edge cases")
    figure("client_outcomes", "Figure 3. Non-standard client trajectory diagnostic for seed 17. Sorting by final accuracy reveals which writers remain difficult. Circle area in the scatter follows client size.")
    p(f"For the chosen clean model at seed 17, client validation accuracy ranges from "
      f"{pct(analysis['client_val_min'])} to {pct(analysis['client_val_max'])}. The median is "
      f"{pct(analysis['client_val_median'])} and the bottom decile is {pct(analysis['client_val_p10'])}. "
      f"Clients are selected between {analysis['participation_min']} and {analysis['participation_max']} times. "
      f"The correlation between selection count and final client accuracy is "
      f"{analysis['participation_accuracy_correlation']:.3f}. This association is descriptive because client difficulty and composition also vary.")
    table([["Observed edge case", "Handling and empirical consequence"],
           ["Rare local classes", f"There are {int((counts == 1).sum()):,} client/class cells with exactly one example. A random local split can place that example on only one side. The code retains the split and the fixed 62-output model."],
           ["Classes missing locally", f"The split has {int((train_counts == 0).sum()):,} zero-count training cells and {int((val_counts == 0).sum()):,} zero-count validation cells out of 6,200. All 62 classes remain represented globally in both splits."],
           ["Small and unequal clients", "All supplied clients have nonempty training and validation sets. Preparation rejects clients with fewer than two examples. FedAvg uses local training counts."],
           ["Partial or failed update round", "Aggregation occurs after all selected updates return. The Ray path aborts on failure and cleans up actors in a finally block. Rerunning recomputes an incomplete run from round zero."]], [135, 375])
    p("The pooled metric can conceal weak performance on individual writers. The trajectory plot makes this "
      "visible, and the restricted-participation stress experiment deliberately removes most writers "
      "from the update process to probe the consequence.")

    page("Parallel client training with Ray Core")
    p("Each round samples exactly four clients. Four actors are created with one CPU and zero GPUs each. "
      "Their train methods are submitted together and retrieved with a single ray.get call. Each actor loads "
      "only its own client file. The selected actors are killed after their updates return. The local cluster "
      "reserves four CPU slots, and each PyTorch process uses one computational thread [3, 4].")
    figure("ray_curves", "Figure 4. Matched serial and Ray runs at C=4%. Curves overlap because client sampling, initialization, minibatch order, local optimizer and aggregation order match.", width=490)
    serial_configs = [Config(clients=4, epochs=chosen["epochs"], seed=s, rounds=rounds) for s in seeds]
    rows = [["Execution", "Mean total seconds", "Mean update seconds/round", "Clean test accuracy (%)"]]
    for label, configs in (("Serial", serial_configs), ("Ray", ray_configs)):
        runs = summaries(configs)
        update_times = [row["update_seconds"] for config in configs
                        for row in read(root / config.name / "history.json")[1:]]
        test_value = pair(accuracy(configs)) if label == "Ray" or all(c.name in tests for c in configs) else "See matched validation"
        rows.append([label, f"{np.mean([r['seconds'] for r in runs]):.1f}", f"{np.mean(update_times):.4f}", test_value])
    table(rows, [70, 110, 165, 165])
    p("Total wall time includes metric evaluation and history writes. Update timing includes actor startup, "
      "weight transfer, training, aggregation and termination. Short client workloads can be dominated by "
      "process startup. Timing was measured on one laptop during this experiment session and should be interpreted "
      "as an observed implementation cost, with contention and warm-up effects.")

    page("Parallel semantics and limiting-case validation")
    figure("ray_timeline", "Figure 5. Non-standard execution timeline from actual actor timestamps in rounds 1-3 of seed 17. Colored groups represent consecutive rounds. Windows are concatenated, omitting startup gaps.")
    ray_check = validation["ray_serial_equivalence"]
    rows = [["Matched round", "Maximum parameter error", "Distinct workers", "Peak concurrent updates"]]
    for row in ray_check["trace"]:
        rows.append([row["round"], f"{row['max_parameter_error']:.2e}", row["distinct_workers"], row["peak_overlapping_updates"]])
    table(rows, [100, 165, 105, 140])
    p("The independent serial/Ray experiment starts both paths from identical weights and samples the same "
      "four clients for three rounds. Acceptance requires maximum parameter error below 1e-6 and four worker "
      "processes per round. All checks pass. Timestamp overlap provides additional evidence that client "
      "methods execute concurrently.")
    table([["Limit or oracle", "Measured result", "Acceptance"],
           ["Unequal counts 1 and 3; local model values 1 and 9", f"Aggregate {validation['unequal_count_oracle']['value']:g}; unweighted value would be 5", "Exactly 7"],
           ["One client, one full-batch SGD step", f"Max gradient-oracle parameter error {validation['single_client_gradient_oracle']['max_parameter_error']:.2e}", "Below 1e-6"],
           ["Ten rounds with learning rate zero and b=0.5", f"Max drift {validation['zero_learning_rate']['max_parameter_error']:.2e}", "Below 1e-6"],
           ["Zero-noise input and local update", "Both differences exactly zero", "Exact equality"]], [205, 205, 100])
    p("The gradient oracle uses an independent direct autograd update on one full client batch. The zero-rate "
      "check still resamples clients and adds noise, so any substantive model change would falsify the "
      "aggregation or optimizer behavior. Its tiny measured drift is floating-point roundoff.")

    page("Laplace perturbation and learning curves")
    p(f"Every training image is transformed as X' = X + noise, with independent Laplace(0,b) entries. "
      f"The sweep uses C=10%, E={noise_epochs}, and the same 100-round schedule as the clean baseline. "
      "For each seed, a fixed perturbed training dataset is created once and reused. Noise is left unbounded. "
      "Labels remain unchanged, and validation and test inputs remain clean.")
    figure("noise_curves", "Figure 6. Accuracy during federated training for six noise scales. Training metrics evaluate the perturbed local inputs used for learning. Validation evaluates clean images. Bands show one sample standard deviation.")
    rows = [["b", "Perturbed train (%)", "Clean validation (%)", "Clean test (%)"]]
    for row in noise_rows:
        rows.append([f"{row['noise']:g}", pair((row['train_acc'], row['train_acc_sd'])),
                     pair((row['val_acc'], row['val_acc_sd'])), pair((row['test_acc'], row['test_acc_sd']))])
    table(rows, [45, 155, 155, 155])
    law = validation["noise_distribution"]
    p(f"At b=0.2, {law['n']:,} sampled pixel perturbations have mean {law['mean']:.6f} "
      f"and variance {law['variance']:.6f}, compared with theoretical variance 2b^2=0.08. "
      f"Adjacent entries have empirical correlation {law['adjacent_correlation']:.4f}. The zero-noise "
      "control reproduces both clean inputs and local model updates exactly.")
    p("Paired runs share initialization, client selections and minibatch orders. A separate noise stream "
      "keeps these controls fixed when b changes. The test stage is executed after validation-based "
      "configuration and noise selection have been written to selection.json.")

    page("Noise choice and privacy interpretation")
    figure("noise_tradeoff", "Figure 7. Final training and clean-test accuracy versus noise scale. Clean training evaluates the same underlying training examples without perturbation. Error bars show one sample standard deviation.")
    p(f"The prespecified utility rule chooses the largest positive b with mean validation accuracy within "
      f"two percentage points of the b=0 baseline ({pct(clean_noise_row['val_acc'])}). It selects "
      f"<b>b={b:g}</b>, with validation {pct(chosen_noise_row['val_acc'])} and test "
      f"{pct(chosen_noise_row['test_acc'])}. The mean test change relative to b=0 is "
      f"{100*(chosen_noise_row['test_acc']-clean_noise_row['test_acc']):+.2f} percentage points. "
      "The small-scale effects should be read alongside the three-seed variation. Large b injects "
      "more pixel variation and visibly reduces useful learning.")
    high_noise = noise_rows[-1]
    p(f"At b=1, accuracy on the fixed perturbed training inputs is {pct(high_noise['train_acc'])}, "
      f"while accuracy on the underlying clean training inputs is {pct(high_noise['clean_train_acc'])} "
      f"and clean test accuracy is {pct(high_noise['test_acc'])}. The model can fit features of the fixed "
      "perturbations that do not transfer to clean inputs. A fresh-noise control would help distinguish "
      "fitting idiosyncratic perturbations from a general clean-input distribution shift.")
    p("For the ideal Laplace mechanism, epsilon = L1 sensitivity / b [5]. Replacing an arbitrary "
      "image in [0,1]^784 has L1 distance at most 784 from its replacement. Under image replacement with a fixed label, "
      f"the conservative bound is therefore epsilon = 784/b, which is {784/b:,.0f} at the selected scale. "
      "A one-pixel adjacency definition gives sensitivity 1 and answers a different privacy question.")
    p("The reported choice preserves utility under the assignment's mechanism. A meaningful deployment "
      "guarantee would require a defined record or client adjacency, protected labels, secret randomness and "
      "accounting for all released metrics and independently perturbed experiment runs. The public seeds "
      "here are for reproducibility. Reusing one fixed noisy dataset across training rounds can be viewed as "
      "post-processing only when later outputs depend exclusively on an appropriately private release.")

    page("Class-specific noise sensitivity")
    figure("class_noise_sensitivity", "Figure 8. Non-standard class diagnostic. Each cell is the change in clean-test recall from b=0, aggregating confusion counts over three seeds. Classes are ordered by their frequency in the supplied local-data collection.", width=480)
    recalls = np.array(analysis["class_recall_by_noise"])
    selected_index = [row["noise"] for row in noise_rows].index(b)
    class_shift = recalls[:, selected_index] - recalls[:, 0]
    worst = int(class_shift.argmin())
    p(f"At b={b:g}, the largest class recall decline is {100*class_shift[worst]:.1f} percentage points "
      f"for class {worst}. Macro-average recall changes from {pct(recalls[:, 0].mean())} to "
      f"{pct(recalls[:, selected_index].mean())}. These class results diagnose where pooled accuracy "
      "can hide uneven degradation. Rare classes have smaller test denominators and greater sampling uncertainty.")

    page("Learning-signal and participation stress experiments")
    figure("validation_experiments", "Figure 9. A capacity check, a stress experiment that restricts participating writers, and a negative control that destroys the image-label link within each client.")
    tiny = validation["tiny_set_memorization"]
    majority = int(train_counts.sum(axis=0).argmax())
    majority_acc = val_counts[:, majority].sum() / val_counts.sum()
    p(f"The tiny balanced subset stays inside client 0 and contains five examples from each of ten locally observed classes. Full-batch SGD with "
      f"learning rate 0.3 reaches {pct(tiny['trace'][-1]['accuracy'])} after 400 steps, exceeding the "
      "98% acceptance threshold. This shows that the chosen network and gradient pipeline can fit a tractable "
      "classification problem. It does not measure generalization.")
    p(f"The shuffled-label control preserves each client's label histogram while permuting its training "
      f"labels. Correct validation labels produce final accuracy {pct(analysis['shuffled_validation_acc'])}, "
      f"compared with {pct(majority_acc)} for the constant predictor using training-majority class {majority}. "
      f"Uniform random guessing would give {pct(1/62)}. The clean run's much higher validation accuracy "
      "depends on the association between images and labels, beyond their marginal frequencies.")
    stress = validation["restricted_participation_stress"]
    p(f"Using only the ten smallest clients for ten rounds gives validation accuracy "
      f"{pct(stress['smallest']['trace'][-1]['val_acc'])}; using only the ten largest gives "
      f"{pct(stress['largest']['trace'][-1]['val_acc'])}. These groups contain "
      f"{stress['smallest']['total_training_samples']:,} and {stress['largest']['total_training_samples']:,} "
      "training examples respectively. Each group uses E=2, seed 17, and all ten members per round. "
      "Sample quantity and writer composition change together, so the difference does not identify a "
      "causal effect of client size. It demonstrates the sensitivity to persistently restricted participation.")
    p("Validation instructions are executable in the repository. Run <font face='Courier'>python -m "
      "fedemnist.validate --ray</font> for the oracles, limits, noise law, capacity experiment and participation "
      "stress. Run <font face='Courier'>python -m fedemnist.validate --shuffled</font> for the 100-round "
      "negative control. Numerical traces and acceptance thresholds are saved in results/validation.json.")

    page("Reproduction, limitations and extended validation")
    p("Download and extract the course ZIP into Assignment3-data. The public repository contains source "
      "code, numerical histories and figures. The source dataset, virtual environment and model checkpoints "
      "are excluded. The exact Python dependency versions are recorded in requirements.lock.")
    p("python3.13 -m venv .venv<br/>source .venv/bin/activate<br/>"
      "python -m pip install -r requirements.txt<br/>python -m fedemnist.prepare<br/>"
      "python -m fedemnist.validate --ray<br/>python -m fedemnist.validate --shuffled<br/>"
      "python -m fedemnist.experiments suite<br/>python -m fedemnist.experiments test<br/>"
      "python -m fedemnist.plots<br/>python -m fedemnist.report", "CodeCustom")
    p("The experiment suite reuses a matching completed run when its checkpoint and history are present. "
      "A fresh clone retrains to recreate the excluded checkpoints before test evaluation. An interrupted "
      "run is recomputed from round zero. Use --output results_reproduced on validation and experiment commands to preserve the included "
      "numerical records. The README explains how each experiment validates a predicted behavior.")
    p("The present scope uses one split, three training seeds, four (C,E) combinations, and a fixed "
      "learning rate. It samples optimization variability within this scope. The 100-round results do not "
      "establish convergence. The test file supplies pooled examples, so writer-specific test generalization "
      "cannot be established from its structure. Local validation supports the client-level diagnostic instead.")
    p("<b>With three additional months</b>, the first month would repeat local splits and training across at "
      "least ten seeds, add writer-held-out evaluation where identifiers permit it, and compare equal "
      "communication, example-processing and wall-time budgets. Client and class bootstrap analyses would "
      "quantify uncertainty in minority outcomes and the noise-selection boundary.")
    p("The second month would calibrate an explicit pixel, image or client privacy objective, include label "
      "protection as needed, use secret randomness and account for repeated releases. Membership-inference "
      "and reconstruction experiments with disjoint attack evaluation data would characterize empirical "
      "exposure alongside the formal analysis. Clipping and fresh-versus-fixed perturbation would be evaluated "
      "as separately declared mechanisms.")
    p("The third month would test Ray on multiple machines with slower clients, bandwidth constraints, "
      "delayed updates, worker crashes and controlled dropout. Experiments would measure update correctness, "
      "resource use and fairness under each condition. Learning-rate schedules and local epoch sweeps would "
      "be chosen on validation before any new held-out evaluation.")

    page("Implementation record and references")
    training_environment = summaries(baseline_configs)[0]["environment"]
    p(f"Execution environment used Python {training_environment['python']}, PyTorch {training_environment['torch']}, "
      f"Ray {training_environment['ray']}, and NumPy {training_environment['numpy']} "
      f"on {training_environment['system']} {training_environment['machine']}. "
      "Training and metric evaluation use deterministic CPU operations. Ray actors reserve one CPU each and "
      "use zero GPUs. Full run histories include selected client IDs, client validation accuracies and update timestamps.")
    if (root / "deepdish/environment.json").exists() and (root / "deepdish/validation.json").exists():
        remote_environment = read(root / "deepdish/environment.json")
        remote_validation = read(root / "deepdish/validation.json")
        remote_error = max(row["max_parameter_error"] for row in remote_validation["ray_serial_equivalence"]["trace"])
        p(f"The checks invoked by <font face='Courier'>python -m fedemnist.validate --ray</font> "
          f"also passed on DeepDish 4 using Python {remote_environment['python']}, "
          f"PyTorch {remote_environment['torch']}, Ray {remote_environment['ray']}, and NumPy {remote_environment['numpy']}. "
          f"It used four CPU actors, one CPU and zero GPUs per actor. The matched serial/Ray maximum parameter "
          f"difference was {remote_error:.2e}. The main experiments and the 100-round shuffled-label control "
          f"used the local CPU environment recorded above.")
    p(f"The source training-file SHA-256 is<br/><font face='Courier' size='7.5'>{manifest['source_sha256']}</font>.")
    table([["Code entry point", "Responsibility"],
           ["fedemnist/prepare.py", "Preprocessing, local split and client file preparation"],
           ["fedemnist/core.py", "Local SGD, sample-weighted aggregation, evaluation and Ray Actors"],
           ["fedemnist/experiments.py", "Three-seed grid, validation selection and separate test stage"],
           ["fedemnist/validate.py", "Controlled algorithm validation and stress experiments"],
           ["fedemnist/plots.py", "Curves and non-standard diagnostic figures"],
           ["fedemnist/report.py", "Reproducible report from recorded results"]], [170, 340])
    refs = [
        ("[1] Caldas et al. (2018). LEAF: A Benchmark for Federated Settings.", "https://arxiv.org/abs/1812.01097"),
        ("[2] McMahan et al. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. AISTATS.", "https://proceedings.mlr.press/v54/mcmahan17a.html"),
        ("[3] Ray Core documentation. Actors.", "https://docs.ray.io/en/latest/ray-core/actors.html"),
        ("[4] Ray Core documentation. Terminating Actors.", "https://docs.ray.io/en/latest/ray-core/actors/terminating-actors.html"),
        ("[5] Dwork and Roth (2014). The Algorithmic Foundations of Differential Privacy. Section 3.3 and Theorem 3.6.", "https://systems.cs.columbia.edu/private-systems-class/papers/Dwork2013Foundations.pdf"),
    ]
    for title, url in refs:
        p(f'{escape(title)} <link href="{url}" color="#14576a">Source</link>.', "SmallCustom")
    p("The course README identifies the supplied data as a subset adapted from LEAF [1]. The server uses "
      "sample-weighted model averaging following FedAvg [2]. Ray's actor APIs supply process execution "
      "and lifecycle control [3, 4]. The Laplace sensitivity relationship follows [5].")
    p("Code and report were prepared with AI assistance. Numerical findings were produced by the included "
      "implementation and the recorded experiments.", "SmallCustom")

    destination = Path("output/pdf/assignment3_report.pdf")
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(str(destination), pagesize=(612, 792),
                                 leftMargin=51, rightMargin=51, topMargin=49, bottomMargin=43,
                                 title="Assignment 3 - Federated Learning and Laplace Input Perturbation",
                                 author="EvelynXuNU")

    def footer(canvas, doc):
        canvas.setStrokeColor(colors.HexColor("#d8e3e7"))
        canvas.line(51, 32, 561, 32)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(muted)
        canvas.drawString(51, 21, "Assignment 3 | Federated EMNIST")
        canvas.drawRightString(561, 21, str(doc.page))

    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print(destination.resolve())


if __name__ == "__main__":
    build()
