# Federated EMNIST and Laplace input perturbation

Assignment 3 implementation for 100 clients and 62 classes. The model is a two-layer fully connected network, `784 -> 128 -> 62`, with ReLU. FedAvg and local SGD are implemented directly in PyTorch. Parallel training uses Ray Core Actors.

## Submission

Upload [`output/pdf/assignment3_report.pdf`](output/pdf/assignment3_report.pdf) to Canvas. The report includes the GitHub username, repository name and public repository link, required plots and discussion, run instructions, references, and a three-month extended validation plan. Page 2 maps each assignment part to its edge cases, predicted and measured validation results, non-standard figures, and rerun commands.

All assignment source code, dependency files, recorded numerical results, and figures are included in this repository. Obtain the course dataset separately from Canvas as described below. The PDF is the Canvas deliverable; source code is submitted through the public repository link in the report.

## Run the experiments

Python 3.13 was used on an Apple Silicon Mac. Training uses CPU, one PyTorch thread per process. Ray reserves four CPUs and zero GPUs. The exact installed environment is recorded in `requirements.lock`.

Download `Assignment3-data.zip` from the course's Canvas Files area and extract it into this directory. The expected files are `Assignment3-data/train_data.npy` and `Assignment3-data/test_data.npy`. These trusted course object arrays require NumPy's `allow_pickle=True` loader. Keep the source data local.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m fedemnist.prepare
python -m fedemnist.validate --ray
python -m fedemnist.validate --shuffled
python -m fedemnist.experiments suite
python -m fedemnist.experiments test
python -m fedemnist.plots
python -m fedemnist.report
```

The report is written to `output/pdf/assignment3_report.pdf`. Figure PNGs appear in `output/figures`. Numerical histories, validation records, selection decisions, and test results appear in `results`. Model checkpoints remain local and are excluded from Git. Completed runs are reused when their configuration and source-data hash match and their checkpoint and history files are present. On a fresh clone, the suite retrains the included numerical runs to create the missing local checkpoints before test evaluation. An interrupted run starts again from round zero and replaces its partial history.

To rerun independently while preserving the included numerical results, use a different output directory for the computational stages. The report and plotting commands use the default `results` directory.

```bash
python -m fedemnist.validate --ray --output results_reproduced
python -m fedemnist.validate --shuffled --output results_reproduced
python -m fedemnist.experiments suite --output results_reproduced
python -m fedemnist.experiments test --output results_reproduced
```

## Experimental protocol

- Invert grayscale with `1-x` so white backgrounds become zero. Each client independently receives a deterministic random split with `floor(0.8*n)` training examples and the remainder for validation. Split seed is 2026, with separate random streams for each client.
- Run 100 communication rounds with SGD, learning rate 0.1, batch size 32, no momentum, and fresh optimizer state for each local update. Sample clients uniformly without replacement within each round. Aggregate their models by local training sample count.
- Compare participation fractions 4% and 10%, crossed with local epoch counts 1 and 3. Repeat with training seeds 17, 29, and 43. Use mean final validation accuracy to select the clean model. Retain the model at round 100.
- For Ray, use four selected client actors in each round. Each actor reads its own client file, receives global weights and returns local weights and sample count. Kill all selected actors after the round. Evaluate global models using aggregate client statistics. In this single-machine simulation, the driver hosts the serial client objects and evaluator. Files and process boundaries simulate client ownership.
- For the noise sweep, use 10% participation and select E from the clean 10% grid. Scales are 0, 0.05, 0.1, 0.2, 0.5, and 1.0. Add independent Laplace noise once to each stored training image and reuse the perturbed dataset across rounds. Leave noise unbounded and leave labels unchanged. Evaluate both the perturbed and clean training views, plus clean validation and test images.
- Use the largest positive scale whose mean final validation accuracy is within two percentage points of the clean baseline as the utility choice. `results/selection.json` is written before the suite runs its Ray models and before test evaluation. The separate `test` stage opens the test data only after selection has been recorded. Test results across scales answer the assignment's requested comparison.
- Use identical initialization, participation and minibatch streams for paired noise runs. Noise has its own stream, so changing b does not change those other random choices. Plot shaded bands and error bars as sample standard deviation across three seeds. These describe training randomness on one fixed split.

## DeepDish access and verification

Access was enabled using the official [MLDS setup utility](https://github.com/numlds/mlds-setup-releases). Campus Wi-Fi or the Northwestern VPN is required. The utility verified key-based SSH on all five DeepDish hosts.

The checks and stress experiments invoked by `python -m fedemnist.validate --ray` also passed on DeepDish 4 with four CPU actors, one CPU and zero GPUs per actor. The 100-round shuffled-label control ran locally. The isolated remote environment reused the server's PyTorch 2.6.0+cu124 and NumPy 1.23.1 and installed Ray 2.60.0 in a project virtual environment. It ran with Python 3.10.12. The CUDA-capable PyTorch build performed this verification on CPU. The exact environment and validation outputs are recorded in `results/deepdish/environment.json` and `results/deepdish/validation.json`.

The main 100-round experiments and test results were produced on the Mac's CPU with the pinned Python 3.13 environment. Remote validation independently checked the algorithm and actor behavior. To connect, use your NetID and an enabled host returned by the setup utility.

```bash
ssh YOUR_NETID@mlds-deepdish4.ads.northwestern.edu
```

## Validation experiments and their interpretation

Run `python -m fedemnist.validate --ray` to execute the following checks and save `results/validation.json`. A failed acceptance rule raises an error.

| Experiment | Prediction and acceptance rule | What it checks |
| --- | --- | --- |
| Unequal-count aggregation oracle | Models 1 and 9 with counts 1 and 3 aggregate to 7 exactly | Distinguishes sample-weighted FedAvg from an unweighted value of 5 |
| Single-client full-batch gradient oracle | One local SGD step matches a direct gradient update to maximum parameter error below 1e-6 | Checks the reduction to ordinary SGD and the learning-rate convention |
| Zero learning rate for ten rounds | Parameters stay within 1e-6 of initialization under client resampling and b=0.5 | Checks that sampling and aggregation cannot create learning independently of gradients |
| Zero-noise identity | Inputs and local updates match exactly at b=0 | Checks that the clean comparison is the limiting noise case |
| Noise law | At b=0.2, absolute sample mean is below 0.005 and variance is within 0.005 of 0.08 | Checks the actual perturbation scale; also records adjacent-pixel correlation |
| Tiny balanced-set memorization | At least 98% training accuracy on 50 examples from client 0 across ten locally observed classes after 400 steps | Checks that architecture, gradients and optimizer can fit a tractable problem |
| Restricted participation stress | Compare global and bottom-decile client validation over ten rounds using only the ten smallest or largest clients | Measures degradation when participation excludes most writers; a descriptive stress experiment |
| Serial versus Ray, matched clients | Maximum parameter difference below 1e-6 over three rounds, with four distinct worker PIDs per round | Tests that parallel execution preserves FedAvg semantics; records real update overlap |

Run `python -m fedemnist.validate --shuffled` for a 100-round negative control. It permutes labels within each client's training set, preserves local label histograms, and keeps validation labels correct. Compare its validation accuracy with the matched clean `C=10%, E=3, seed=17` run. Learning should collapse toward an empirical frequency baseline. Uniform chance is 1/62, while label imbalance makes a majority-class baseline more appropriate.

The non-standard figures expose client accuracy over time, actual actor execution overlap, and class-level recall changes under noise ranked by class frequency. These diagnose unequal benefits, real concurrency, and minority-class degradation that pooled accuracy can conceal.

Run `python -m unittest discover -s tests` for the reproduction regression. It removes a checkpoint from a completed synthetic-data run to simulate a public clone, checks that retraining restores the same weights, checks reuse of a complete run, and checks that an interrupted replacement leaves no completion marker. This software regression complements the algorithm validation experiments above.

## Privacy interpretation

The experiments follow the assignment's input-noise mechanism. The formal Laplace calibration is `epsilon = L1 sensitivity / b`. Replacing an arbitrary 784-pixel image in `[0,1]^784` has sensitivity at most 784, so image-level epsilon can be as large as `784/b` under a fixed-label replacement definition. Pixel adjacency gives a different sensitivity. Full record or client replacement requires additional analysis, including label protection.

Noise seeds are public to make these experiments reproducible. Such deterministic noise and the released unprivatized metrics support a utility study. A deployed privacy guarantee would require secret randomness, a defined threat model, label handling, and accounting for every released result. Independent noise releases for the same record also compose. These experiments do not estimate an attack-based privacy guarantee.

## Sources

- Caldas et al., *LEAF: A Benchmark for Federated Settings* (2018), <https://arxiv.org/abs/1812.01097>.
- McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data* (2017), <https://proceedings.mlr.press/v54/mcmahan17a.html>.
- Ray Core Actors, <https://docs.ray.io/en/latest/ray-core/actors.html>.
- Ray actor termination, <https://docs.ray.io/en/latest/ray-core/actors/terminating-actors.html>.
- Dwork and Roth, *The Algorithmic Foundations of Differential Privacy* (2014), section 3.3, <https://systems.cs.columbia.edu/private-systems-class/papers/Dwork2013Foundations.pdf>.

Code and report were prepared with AI assistance. All reported numerical results are computed by the included implementation.
