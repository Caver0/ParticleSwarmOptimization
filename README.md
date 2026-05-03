# ParticleSwarmOptimization

PSO project to compare different ways of evaluating fitness without touching the core of the algorithm. The idea behind the project is simple: to be able to run reproducible executables, save results in JSON, and then turn them into tables and plots that help understand how the program behaves.

## Project structure

```text
ParticleSwarmOptimization/
├── src/
│   └── pso_lab/
│       ├── core/
│       │   ├── config.py          # Parameters of a PSO execution
│       │   ├── models.py          # Auxiliary models and timing metrics
│       │   ├── boundaries.py      # Search space boundary management
│       │   └── optimizer.py       # Main implementation of the algorithm
│       ├── objectives/
│       │   ├── base.py            # Common interface for objective functions
│       │   ├── benchmarks.py      # Sphere, Rosenbrock, Rastrigin and Ackley
│       │   └── __init__.py        # build_objective(...) factory
│       ├── parallel/
│       │   ├── evaluators.py          # Sequential, threading and multiprocessing evaluation
│       │   ├── async_evaluator.py     # Asyncio evaluation with optional artificial delay
│       │   └── vectorized_evaluator.py # NumPy batch evaluation
│       ├── experiments/
│       │   ├── runner.py          # Runs a complete execution with the selected mode
│       │   ├── summary.py         # Summarizes multiple executions into means, minimums and maximums
│       │   └── pyswarm_runner.py  # Adapter to compare against pyswarm
│       ├── io/
│       │   ├── logging_utils.py   # Log configuration
│       │   └── results.py         # Saving results and summaries in JSON
│       ├── viz/
│       │   └── plots.py           # Convergence, timing and trajectory plots
│       └── cli.py                 # Shared arguments for the scripts
├── tests/                         # Tests for optimizer, evaluators, plots and persistence
├── results/                       # Raw benchmark and comparison results
├── reports/
│   └── plots/                     # Figures generated from the results
├── logs/                          # Timestamped execution logs
├── run_pso.py                     # Single PSO execution
├── run_benchmarks.py              # Comparison between evaluation modes
├── run_grid_search.py             # Hyperparameter sweep for w, c1 and c2
├── run_best_configs_comparison.py # Re-runs the best configurations found
├── run_pyswarm_baseline.py        # Comparison between this implementation and pyswarm
├── analyze_results.py             # Reads results/ and generates tables and plots
├── make_viz.py                    # Generates visualizations of particle movement
├── _repo_bootstrap.py             # Makes src/ visible when running from the project root
├── requirements.txt               # Project dependencies
└── README.md
```

## What each part does

`src/pso_lab/core` is the core of the project. It contains the optimizer, the configuration of each execution, and the boundary logic.

`src/pso_lab/objectives` groups the objective functions. They are separated from the optimizer so that the algorithm can be reused for other problems without mixing swarm logic with benchmark logic.

`src/pso_lab/parallel` encapsulates fitness evaluation. This separation is important because it allows comparing `sequential`, `threading`, `multiprocessing`, `asyncio`, and `vectorized` without rewriting the rest of the algorithm.

`src/pso_lab/experiments` is the layer that connects everything: it launches complete executions, summarizes multiple seeds, and prepares the comparison against `pyswarm`.

`src/pso_lab/io` handles logs and persistence. JSON was chosen for results because it is easy to inspect manually, easy to version, and enough for the amount of data handled in the project.

`src/pso_lab/viz` turns results into useful plots.

`tests/` covers the important pieces so that the experiments do not become a black box that is hard to trust, and it also makes code validation easier.

## What each executable provides

`run_pso.py` is the minimum execution. It is used to quickly check that the algorithm converges, that the history is saved correctly, and that the basic pipeline is healthy. It now accepts `--mode`, so the same entrypoint can run `sequential`, `threading`, `multiprocessing`, `asyncio`, or `vectorized`.

`run_benchmarks.py` is the most direct experiment to compare evaluation modes while keeping everything else fixed. The important thing here is not to search for the best configuration, but to isolate the evaluator cost.

`run_grid_search.py` explores combinations of `w`, `c1`, and `c2`. These parameters were chosen because they have the greatest impact on the balance between exploration and exploitation. The search was not expanded much further in order to avoid turning the experiment into something huge and hard to interpret.

`run_best_configs_comparison.py` takes the winning configurations from the grid search and runs them again under equal conditions. It is the script that best shows whether the evaluator only changes execution time or also affects final quality.

`run_pyswarm_baseline.py` compares the project implementation against an external reference. Useful to see whether it really adds value.

`analyze_results.py` and `make_viz.py` close the workflow. One summarizes and plots.

## V3 Asyncio

V3 adds an `asyncio` evaluator that keeps the same contract as the other evaluators: the PSO core still calls `evaluate(objective_function, positions)` and still receives an `np.ndarray` of fitness values. The difference is only in how the fitness evaluations are scheduled.

This mode is designed for I/O-bound or latency-bound scenarios. A configurable artificial delay per particle was added to simulate cases such as API calls, database queries, or external services. The delay is sampled in a reproducible way from the evaluator itself, so the objective interface does not need to become asynchronous.

For purely CPU-bound objectives, no speedup is expected. `asyncio` does not reduce the numerical computation cost of `Sphere`, `Rastrigin`, `Rosenbrock`, or `Ackley`; it only allows cooperative overlap when the evaluation includes waiting time.

Useful examples:

```bash
python run_pso.py --objective sphere --dimension 2 --mode asyncio --seed 42
python run_benchmarks.py --modes asyncio --dimensions 2 10 --objectives sphere rastrigin --seeds 0 1
python run_benchmarks.py --modes asyncio --dimensions 2 --objectives sphere --seeds 0 1 --async-min-delay 0.01 --async-max-delay 0.05
python run_grid_search.py --mode asyncio --dimensions 2 --objectives sphere --seeds 0
```

When `--async-min-delay` and `--async-max-delay` are both `0.0`, the evaluator behaves as a no-delay async baseline. When they are greater than zero, each particle gets its own sampled delay, which makes the benchmark more representative of asymmetric waiting times.

## V4 Vectorized / NumPy

V4 adds a `vectorized` evaluator that keeps the same public contract as the other evaluators: `evaluate(objective_function, positions) -> np.ndarray`. Instead of creating explicit workers, it tries to evaluate the whole swarm through `objective_function.evaluate_batch(positions)` when that batch API is available.

This is a form of implicit parallelism through NumPy. The work is still launched from one Python process, but most of the heavy numerical operations move into optimized array kernels. For cheap and vectorizable objectives, this can be more efficient than `threading` or `multiprocessing` because it avoids Python loops, thread coordination, process startup, IPC, and serialization overhead.

If an objective does not implement `evaluate_batch`, `VectorizedEvaluator` falls back automatically to a compatible sequential loop. That means V4 does not require a new PSO core and does not break custom objectives that only implement `__call__`.

Useful examples:

```bash
python run_pso.py --objective sphere --dimension 2 --mode vectorized --seed 42
python run_benchmarks.py --modes vectorized --dimensions 2 10 30 --objectives sphere rastrigin --seeds 0 1
python run_grid_search.py --mode vectorized --dimensions 2 --objectives sphere --seeds 0 1
```

## Results obtained

### Benchmarks across modes

The overall picture is quite clear: with the functions used in this project, `sequential` wins in execution time very consistently. In the results saved in `best_config_comparison`, the average summary is as follows:

| Mode | Average time per case (s) | Times it was the fastest | Final quality |
| --- | ---: | ---: | --- |
| `sequential` | 0.012 | 12/12 | Same as the others |
| `threading` | 0.037 | 0/12 | Same as the others |
| `multiprocessing` | 0.261 | 0/12 | Same as the others |

In addition, across the 12 analyzed combinations of objective and dimension, `sequential` was always the fastest mode. This matches what could be expected: here the objective functions are lightweight, and the extra cost of coordinating threads or processes outweighs any possible gain.

What is interesting is that final quality does not change depending on the evaluator when the same configuration is kept. In `best_config_comparison`, the 12 comparisons produced the same `mean_best_value` across modes.

### Hyperparameter grid search

The grid search leaves a useful conclusion: there is no single good configuration for everything. In low dimensions, several good solutions appear with more conservative parameters, often with `w=0.4` and moderate coefficients. On the other hand, when moving up to `d=10` or `d=30`, configurations with `w=0.7` and stronger social or cognitive influence start to appear, depending on the function.

This makes quite a lot of sense. In simple problems, the swarm does not need that much inertia to move well. In larger problems or harsher landscapes, an overly timid configuration falls short.

That is why `run_best_configs_comparison.py` stores one best configuration per objective and dimension. It is not an arbitrary decision: it comes directly from the sweep results.

| Dimension | Objective | `w` | `c1` | `c2` |
| ---: | --- | ---: | ---: | ---: |
| 2 | `sphere` | 0.4 | 1.0 | 1.0 |
| 2 | `rosenbrock` | 0.4 | 1.5 | 1.5 |
| 2 | `rastrigin` | 0.4 | 1.5 | 1.0 |
| 2 | `ackley` | 0.4 | 1.0 | 1.0 |
| 10 | `sphere` | 0.4 | 1.5 | 2.0 |
| 10 | `rosenbrock` | 0.7 | 1.5 | 1.5 |
| 10 | `rastrigin` | 0.7 | 1.5 | 1.5 |
| 10 | `ackley` | 0.7 | 2.0 | 1.0 |
| 30 | `sphere` | 0.7 | 2.0 | 1.0 |
| 30 | `rosenbrock` | 0.7 | 2.0 | 1.0 |
| 30 | `rastrigin` | 0.4 | 2.0 | 1.0 |
| 30 | `ackley` | 0.7 | 2.0 | 1.0 |

### Comparison against pyswarm

Compared to `pyswarm`, the reading is a bit more nuanced. In terms of quality, there is no absolute winner. Taking the 12 combinations of objective and dimension used in the comparison, the project implementation achieves a better `mean_best_value` in 6 cases, while `pyswarm` performs better in the other 6.

In terms of time, however, things change quite a lot:

| Project solver | Cases with better `mean_best` than `pyswarm` | Cases faster than `pyswarm` | Total cases |
| --- | ---: | ---: | ---: |
| `sequential` | 6 | 12 | 12 |
| `threading` | 6 | 1 | 12 |
| `multiprocessing` | 6 | 0 | 12 |

The practical conclusion is that `pyswarm` is still a useful reference, but in this particular scenario it does not outperform the project implementation. And even when quality is split evenly, the project version has one important advantage: more control over the experiment, more traceability, and better inspection capabilities.

### What the visualizations add

Convergence plots and particle movement plots help put a face to the numbers. In `Sphere`, convergence looks cleaner and more direct; in `Rastrigin` and `Ackley`, the trajectory is noticeably less smooth, which fits the presence of many local minima or trickier regions. They do not replace tables, but they help a lot when interpreting them.

## Short and honest reading

If the project had to be summarized in one simple idea, it would be this: for lightweight, CPU-bound objective functions, parallelizing in Python does not always help, and sometimes it makes total execution time significantly worse. In this repository, with this workload, `sequential` is the most reasonable option.

That does not make `threading`, `multiprocessing`, `asyncio`, or `vectorized` useless. It simply means that here they were not playing on favorable ground. If the fitness were much more expensive, if there were blocking calls, or if the evaluation relied more heavily on libraries that release the GIL, the story could change.

## Usage note

The scripts at the project root can be run directly from VS Code using the Run button. Each one includes an `if __name__ == "__main__"` block with editable arguments, so there is no need to type the full command in the terminal.

To install the required dependencies:

```bash
pip install -r requirements.txt
```

## Repository

Project source code and tracking:
[GitHub - ParticleSwarmOptimization](https://github.com/Caver0/ParticleSwarmOptimization)
