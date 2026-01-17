# PySATL-MPEST benchmarks

This directory contains the performance test suite for **PySATL-MPEST**, built using [Airspeed Velocity (asv)](https://asv.readthedocs.io/).

## Usage

Since this project uses Poetry, ensure dependencies are installed (including `asv`):

```
poetry install
```

### 1. Configure Machine
Before running benchmarks for the first time, register your machine's information.

```
# Make sure you are inside the poetry shell or use 'poetry run'
poetry run asv machine
```

### 2. Quick Run (Development Mode)
If you are writing code and want to quickly test the performance of your current changes without creating a separate virtual environment or committing code:

Note: This mode uses your current local environment. By default, results from `--quick` runs are not saved.
```
# Run benchmarks only once
poetry run asv run --python=same --quick

# Run all benchmarks
poetry run asv run --python=same

# Run only distributions benchmarks
poetry run asv run --python=same --bench bench_distributions

# Run only the DistributionMethods class inside bench_distributions
poetry run asv run --python=same --bench DistributionMethods
```

### 3. Regression Testing (Compare Branches)
To compare the performance of your current branch (`HEAD`) against the `main` branch to see if your changes made things faster or slower:
Results are saved for comparison.

```
# Compare HEAD (current state) vs refactor/rework-arch branch
poetry run asv continuous refactor/rework-arch HEAD
```

### 4. Full History Run
To run benchmarks across the history of commits (configured in asv.conf.json). This creates virtual environments for each commit to ensure accuracy.
```
# Run all commits (might take a long time!)
poetry run asv run

# Run only recent changes (e.g., last 10 commits)
poetry run asv run HEAD~10..HEAD
```

### 5. Results

For greater control, a graphical view, and to have results saved for future comparison you can run ASV commands (record results and generate HTML):
```
# 1. Generate HTML report
poetry run asv publish

# 2. Serve the report locally
poetry run asv preview
```
This will open the results in your default browser at `http://127.0.0.1:8080`.
