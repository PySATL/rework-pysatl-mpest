import os
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import skfuzzy as fuzz
from scipy.stats import entropy
from sklearn.cluster import KMeans

from rework_pysatl_mpest import Exponential, MixtureModel
from rework_pysatl_mpest.estimators.iterative.breakpointers.step_breakpointer import StepBreakpointer
from rework_pysatl_mpest.estimators.iterative.pipeline import Pipeline
from rework_pysatl_mpest.estimators.iterative.steps.block import MaximizationStrategy, OptimizationBlock
from rework_pysatl_mpest.estimators.iterative.steps.expectation_step import ExpectationStep
from rework_pysatl_mpest.estimators.iterative.steps.maximization_step import MaximizationStep
from rework_pysatl_mpest.Initializers.clusterize_initializer import ClusterizeInitializer
from rework_pysatl_mpest.Initializers.strategies import ClusterMatchStrategy, EstimationStrategy
from rework_pysatl_mpest.optimizers.scipy_powell import ScipyPowell

plt.style.use("default")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = [12, 8]
plt.rcParams["font.size"] = 12
plt.rcParams["font.weight"] = "bold"

script_dir = os.path.dirname(os.path.abspath(__file__))
results_path = os.path.join(script_dir, "em_pipeline_results")
os.makedirs(results_path, exist_ok=True)


class CMeans:
    def __init__(self, n_clusters, m=2.0, error=0.005, maxiter=1000):
        self.n_clusters = n_clusters
        self.m = m
        self.error = error
        self.maxiter = maxiter
        self.u_ = None
        self.cluster_centers_ = None

    def fit_transform(self, X):
        X_reshaped = X.T.astype(np.float64)
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            X_reshaped, self.n_clusters, self.m, error=self.error, maxiter=self.maxiter
        )
        self.u_ = u.T
        self.cluster_centers_ = cntr
        return self.u_


def create_random_initial_mixture():
    return MixtureModel(
        [
            Exponential(loc=np.random.uniform(0, 5), rate=np.random.uniform(0.1, 2.0)),
            Exponential(loc=np.random.uniform(3, 8), rate=np.random.uniform(0.1, 2.0)),
            Exponential(loc=np.random.uniform(6, 12), rate=np.random.uniform(0.1, 2.0)),
        ],
        [1 / 3, 1 / 3, 1 / 3],
    )


def create_true_mixture():
    return MixtureModel(
        [Exponential(loc=2.25, rate=1.0), Exponential(loc=2.5, rate=0.5), Exponential(loc=2.75, rate=0.67)],
        [1 / 3, 1 / 3, 1 / 3],
    )


def clusterize_initialization(X: np.ndarray, mixture: MixtureModel, is_accurate: bool, is_soft: bool, clusterizer: Any):
    initializer = ClusterizeInitializer(is_accurate=is_accurate, is_soft=is_soft, clusterizer=clusterizer)
    mixture = initializer.perform(
        X=X.reshape(-1, 1),
        dists=mixture.components,
        cluster_match_strategy=ClusterMatchStrategy.AKAIKE,
        estimation_strategies=[
            EstimationStrategy.QFUNCTION,
            EstimationStrategy.QFUNCTION,
            EstimationStrategy.QFUNCTION,
        ],
    )
    return mixture


def calculate_kl_divergence(mixture1, mixture2, x_min=0.001, x_max=25, n_points=1000):
    x = np.linspace(x_min, x_max, n_points)
    pdf1 = np.array([mixture1.pdf(xi) for xi in x])
    pdf2 = np.array([mixture2.pdf(xi) for xi in x])
    epsilon = 1e-10
    pdf1 = np.clip(pdf1, epsilon, None)
    pdf2 = np.clip(pdf2, epsilon, None)
    pdf1 = pdf1 / np.sum(pdf1)
    pdf2 = pdf2 / np.sum(pdf2)
    kl_div = entropy(pdf1, pdf2)
    return kl_div


def calculate_log_likelihood(mixture, X):
    try:
        return float(mixture.loglikelihood(X))
    except Exception as e:
        print(f"Ошибка при вычислении правдоподобия: {e}")
        return -np.inf


def create_simple_em_pipeline(max_iterations=1000):
    e_step = ExpectationStep(is_soft=True)
    blocks = [
        OptimizationBlock(
            component_id=0, params_to_optimize={"loc", "rate"}, maximization_strategy=MaximizationStrategy.QFUNCTION
        ),
        OptimizationBlock(
            component_id=1, params_to_optimize={"loc", "rate"}, maximization_strategy=MaximizationStrategy.QFUNCTION
        ),
        OptimizationBlock(
            component_id=2, params_to_optimize={"loc", "rate"}, maximization_strategy=MaximizationStrategy.QFUNCTION
        ),
    ]
    optimizer = ScipyPowell()
    m_step = MaximizationStep(blocks=blocks, optimizer=optimizer)
    steps = [e_step, m_step]
    breakpointers = [StepBreakpointer(max_steps=max_iterations)]
    pipeline = Pipeline(
        steps=steps,
        breakpointers=breakpointers,
        pruners=None,
    )
    return pipeline


def run_em_pipeline_experiment(X, initial_mixture, true_mixture, max_iterations=10000):
    pipeline = create_simple_em_pipeline(max_iterations)
    start_time = time.time()
    final_mixture = initial_mixture
    try:
        final_mixture = pipeline.fit(X, final_mixture)
        execution_time = time.time() - start_time
        final_ll = calculate_log_likelihood(final_mixture, X)
        convergence_info = {
            "total_iterations": len(pipeline.logger),
            "final_log_likelihood": final_ll,
        }
        kl_divergence = calculate_kl_divergence(final_mixture, true_mixture)
        return {
            "final_mixture": final_mixture,
            "execution_time": execution_time,
            "kl_divergence": kl_divergence,
            "convergence_info": convergence_info,
        }
    except Exception as e:
        print(f"Ошибка при выполнении EM: {e}")
        return None


def run_comprehensive_pipeline_experiment(n_samples=1000, max_iterations=10000, run_id=0):
    true_mixture = create_true_mixture()
    random_mixture = create_random_initial_mixture()
    X = true_mixture.generate(n_samples)

    initialization_methods = {"EM": random_mixture, "Kmeans": None, "Cmeans": None}

    kmeans_init_time = 0
    try:
        kmeans = KMeans(n_clusters=true_mixture.n_components, n_init=10)
        start_time = time.time()
        initialization_methods["Kmeans"] = clusterize_initialization(
            X, random_mixture, is_accurate=False, is_soft=False, clusterizer=kmeans
        )
        kmeans_init_time = time.time() - start_time
    except Exception as e:
        print(f"Ошибка KMeans инициализации: {e}")

    fuzzy_init_time = 0
    try:
        cmeans = CMeans(n_clusters=random_mixture.n_components)
        start_time = time.time()
        initialization_methods["Cmeans"] = clusterize_initialization(
            X, random_mixture, is_accurate=False, is_soft=True, clusterizer=cmeans
        )
        fuzzy_init_time = time.time() - start_time
    except Exception as e:
        print(f"Ошибка CMeans инициализации: {e}")

    metrics_data = []
    final_mixtures = {}

    for method_name, initial_mixture in initialization_methods.items():
        if initial_mixture is not None:
            result = run_em_pipeline_experiment(X, initial_mixture, true_mixture, max_iterations)
            if result is not None:
                init_time = 0
                if method_name == "Kmeans":
                    init_time = kmeans_init_time
                elif method_name == "Cmeans":
                    init_time = fuzzy_init_time

                metrics_data.append(
                    {
                        "method": method_name,
                        "n_samples": n_samples,
                        "run_id": run_id,
                        "kl_divergence": result["kl_divergence"],
                        "final_log_likelihood": result["convergence_info"]["final_log_likelihood"],
                        "total_iterations": result["convergence_info"]["total_iterations"],
                        "execution_time": result["execution_time"],
                        "initialization_time": init_time,
                        "total_time": result["execution_time"] + init_time,
                    }
                )
                final_mixtures[method_name] = result["final_mixture"]

    return metrics_data, X, true_mixture, final_mixtures


def plot_comparison_results(aggregated_df_for_n, X_sample, true_mixture, final_mixtures, n_samples):
    if aggregated_df_for_n.empty:
        print(f"Нет данных для визуализации n_samples={n_samples}")
        return

    print(f"Создание графиков для {n_samples} samples...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    x_plot = np.linspace(0, 10, 1000)

    ax1.hist(X_sample, bins=50, density=True, alpha=0.3, color="gray", edgecolor="black", label="Данные")

    true_pdf = np.array([true_mixture.pdf(xi) for xi in x_plot])
    ax1.plot(x_plot, true_pdf, "k-", linewidth=3, label="Истинное распределение", alpha=0.8)

    colors = {"EM": "red", "Kmeans": "blue", "Cmeans": "green"}
    for method_name, final_mixture in final_mixtures.items():
        if final_mixture is not None:
            fitted_pdf = np.array([final_mixture.pdf(xi) for xi in x_plot])
            ax1.plot(
                x_plot,
                fitted_pdf,
                "--",
                linewidth=2,
                color=colors.get(method_name, "black"),
                label=f"{method_name} распределение",
            )

    ax1.set_xlabel("Значение")
    ax1.set_ylabel("Плотность")
    ax1.set_title(f"Сравнение методов инициализации - {n_samples} samples")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 10)

    ax2.axis("off")
    metrics_data = []
    for _, row in aggregated_df_for_n.iterrows():
        method = row["method"]
        kl_mean = row["kl_divergence_mean"]
        kl_std = row["kl_divergence_std"]
        time_mean = row["total_time_mean"]
        time_std = row["total_time_std"]
        metrics_data.append([method, f"{kl_mean:.4f} ± {kl_std:.4f}", f"{time_mean:.3f} ± {time_std:.3f}s"])

    if metrics_data:
        table = ax2.table(
            cellText=metrics_data,
            colLabels=["Method", "KL Div (mean ± std)", "Total Time (mean ± std)"],
            cellLoc="center",
            loc="center",
            bbox=[0, 0, 1, 1],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        for j in range(len(metrics_data[0])):
            table[(0, j)].set_facecolor("#4B8BBE")
            table[(0, j)].set_text_props(weight="bold", color="white")

    plt.tight_layout()

    filename = os.path.join(results_path, f"comparison_{n_samples}.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"График сохранен как: {filename}")
    plt.close()


if __name__ == "__main__":
    np.random.seed(12)

    sample_sizes = [100, 200, 300, 400, 500, 1000]
    N_RUNS = 5
    all_metrics = []

    for n_samples in sample_sizes:
        print(f"\nEXPERIMENT FOR {n_samples} SAMPLES")
        for run_id in range(1, N_RUNS + 1):
            print(f"Запуск {run_id}/{N_RUNS}")
            metrics_data, X, true_mixture, final_mixtures = run_comprehensive_pipeline_experiment(
                n_samples, max_iterations=500, run_id=run_id
            )
            if metrics_data:
                all_metrics.extend(metrics_data)

    if all_metrics:
        results_df = pd.DataFrame(all_metrics)
        results_df.to_csv(os.path.join(results_path, "em_pipeline_comparison_raw_results.csv"), index=False)

        agg_metrics = ["kl_divergence", "final_log_likelihood", "total_iterations", "execution_time", "total_time"]
        aggregated_data = []

        for n_samples in sample_sizes:
            for method in ["EM", "Kmeans", "Cmeans"]:
                subset = results_df[(results_df["n_samples"] == n_samples) & (results_df["method"] == method)]
                if not subset.empty:
                    row = {"method": method, "n_samples": n_samples}
                    for metric in agg_metrics:
                        row[f"{metric}_mean"] = subset[metric].mean()
                        row[f"{metric}_std"] = subset[metric].std()
                    aggregated_data.append(row)

        aggregated_df = pd.DataFrame(aggregated_data)
        aggregated_df.to_csv(os.path.join(results_path, "em_pipeline_comparison_aggregated_results.csv"), index=False)

        print("\nAGGREGATED EXPERIMENTS RESULTS")
        print(aggregated_df.to_string(index=False))

        for n_samples in sample_sizes:
            agg_for_n = aggregated_df[aggregated_df["n_samples"] == n_samples]
            true_mixture_temp = create_true_mixture()
            X_temp = true_mixture_temp.generate(n_samples)

            _, _, _, final_mixtures_temp = run_comprehensive_pipeline_experiment(
                n_samples, max_iterations=500, run_id=0
            )

            plot_comparison_results(agg_for_n, X_temp, true_mixture_temp, final_mixtures_temp, n_samples)

    print("\nExperiments completed!")
    print("Results saved in folder: em_pipeline_results/")
