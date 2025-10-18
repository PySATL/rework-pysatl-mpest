import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import skfuzzy as fuzz
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score

from rework_pysatl_mpest import Exponential, MixtureModel
from rework_pysatl_mpest.Initializers.clusterize_initializer import ClusterizeInitializer
from rework_pysatl_mpest.Initializers.strategies import ClusterMatchStrategy, EstimationStrategy

MIN_LABELS_COUNT = 2
SAMPLES_ALL_COMPARISON = 500
N_EXPERIMENTS = 5

script_dir = os.path.dirname(os.path.abspath(__file__))
results_path = os.path.join(script_dir, "results")
os.makedirs(results_path, exist_ok=True)

plt.style.use("default")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = [12, 8]
plt.rcParams["font.size"] = 12
plt.rcParams["font.weight"] = "bold"


def save_table_as_image(df, filename, title):
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.axis("tight")
    ax.axis("off")

    formatted_data = []
    for i, row in df.iterrows():
        formatted_row = []
        for col in df.columns:
            value = row[col]
            if isinstance(value, str) and "±" in value:
                formatted_row.append(value)
            else:
                formatted_row.append(f"{value:.4f}" if isinstance(value, (int, float)) else str(value))
        formatted_data.append(formatted_row)

    table = ax.table(
        cellText=formatted_data,
        rowLabels=df.index,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(18)
    table.scale(1.5, 2.0)
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor("#4B8BBE")
        table[(0, i)].set_text_props(weight="bold", color="white", size=18)
    for i in range(len(df.index)):
        table[(i + 1, -1)].set_facecolor("#F0F0F0")
        table[(i + 1, -1)].set_text_props(weight="bold", size=18)
    plt.title(title, fontsize=24, fontweight="bold", pad=30)
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, f"{filename}.png"), dpi=300, bbox_inches="tight")
    plt.close()


def save_presentation_table(df, filename, title):
    fig, ax = plt.subplots(figsize=(24, 16))
    ax.axis("tight")
    ax.axis("off")

    formatted_data = []
    for i, row in df.iterrows():
        formatted_row = []
        for col in df.columns:
            value = row[col]
            if isinstance(value, str) and "±" in value:
                formatted_row.append(value)
            else:
                formatted_row.append(f"{value:.4f}" if isinstance(value, (int, float)) else str(value))
        formatted_data.append(formatted_row)

    table = ax.table(
        cellText=formatted_data,
        rowLabels=df.index,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(20)
    table.scale(1.8, 2.5)
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor("#4B8BBE")
        table[(0, i)].set_text_props(weight="bold", color="white", size=20)
    for i in range(len(df.index)):
        table[(i + 1, -1)].set_facecolor("#F0F0F0")
        table[(i + 1, -1)].set_text_props(weight="bold", size=20)
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            table[(i + 1, j)].set_text_props(size=18)
    plt.title(title, fontsize=28, fontweight="bold", pad=40)
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, f"{filename}.png"), dpi=300, bbox_inches="tight")
    plt.close()


def save_multiindex_table_as_image(df, filename, title):
    fig, ax = plt.subplots(figsize=(24, 16))
    ax.axis("tight")
    ax.axis("off")
    columns = [f"{col[0]}\n{col[1]}" for col in df.columns] if isinstance(df.columns, pd.MultiIndex) else df.columns

    formatted_data = []
    for i, row in df.iterrows():
        formatted_row = []
        for col in df.columns:
            value = row[col]
            if isinstance(value, str) and "±" in value:
                formatted_row.append(value)
            else:
                formatted_row.append(f"{value:.4f}" if isinstance(value, (int, float)) else str(value))
        formatted_data.append(formatted_row)

    table = ax.table(
        cellText=formatted_data,
        rowLabels=df.index,
        colLabels=columns,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(16)
    table.scale(1.5, 2.0)
    for i in range(len(columns)):
        table[(0, i)].set_facecolor("#4B8BBE")
        table[(0, i)].set_text_props(weight="bold", color="white", size=16)
    for i in range(len(df.index)):
        table[(i + 1, -1)].set_facecolor("#F0F0F0")
        table[(i + 1, -1)].set_text_props(weight="bold", size=16)
    plt.title(title, fontsize=22, fontweight="bold", pad=30)
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, f"{filename}.png"), dpi=300, bbox_inches="tight")
    plt.close()


def format_with_std(mean_value, std_value):
    if pd.isna(mean_value) or pd.isna(std_value):
        return f"{mean_value:.4f}"
    return f"{mean_value:.4f} ± {std_value:.4f}"


def calculate_clustering_metrics(X, labels):
    if len(np.unique(labels)) < MIN_LABELS_COUNT:
        return np.nan, np.nan
    X_reshaped = X.reshape(-1, 1)
    silhouette = silhouette_score(X_reshaped, labels)
    db = davies_bouldin_score(X_reshaped, labels)
    return silhouette, db


def calculate_kl_divergence(mixture2, mixture1, x_min=0.001, x_max=25, n_points=1000):
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


def plot_comparison_kmeans(X, true_mixture, initial_mixture, mixtures_dict, title, filename):
    fig, ax = plt.subplots(figsize=(16, 10))
    x_plot = np.linspace(0, 25, 1000)
    ax.hist(X, bins=50, density=True, alpha=0.3, color="gray", edgecolor="black", label="Data")
    ax.plot(x_plot, [true_mixture.pdf(xi) for xi in x_plot], "r-", linewidth=4, label="True", alpha=0.8)
    ax.plot(x_plot, [initial_mixture.pdf(xi) for xi in x_plot], "b--", linewidth=4, label="Initial", alpha=0.8)

    colors = ["green", "magenta", "cyan", "orange"]
    linestyles = ["-", "--", "-.", ":"]
    for i, (method_name, mixture) in enumerate(mixtures_dict.items()):
        if mixture is not None:
            ax.plot(
                x_plot,
                [mixture.pdf(xi) for xi in x_plot],
                color=colors[i % len(colors)],
                linestyle=linestyles[i % len(linestyles)],
                linewidth=3,
                label=method_name,
                alpha=0.8,
            )
    ax.set_xlabel("Value", fontweight="bold", fontsize=14)
    ax.set_ylabel("Density", fontweight="bold", fontsize=14)
    ax.set_title(title, fontweight="bold", fontsize=18)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 25)
    ax.tick_params(axis="both", which="major", labelsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, f"{filename}.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_comparison_cmeans(X, true_mixture, initial_mixture, mixtures_dict, title, filename):
    fig, ax = plt.subplots(figsize=(16, 10))
    x_plot = np.linspace(0, 25, 1000)
    ax.hist(X, bins=50, density=True, alpha=0.3, color="gray", edgecolor="black", label="Data")
    ax.plot(x_plot, [true_mixture.pdf(xi) for xi in x_plot], "r-", linewidth=4, label="True", alpha=0.8)
    ax.plot(x_plot, [initial_mixture.pdf(xi) for xi in x_plot], "b--", linewidth=4, label="Initial", alpha=0.8)

    colors = ["green", "magenta", "cyan", "orange"]
    linestyles = ["-", "--", "-.", ":"]
    for i, (method_name, mixture) in enumerate(mixtures_dict.items()):
        if mixture is not None:
            ax.plot(
                x_plot,
                [mixture.pdf(xi) for xi in x_plot],
                color=colors[i % len(colors)],
                linestyle=linestyles[i % len(linestyles)],
                linewidth=3,
                label=method_name,
                alpha=0.8,
            )
    ax.set_xlabel("Value", fontweight="bold", fontsize=14)
    ax.set_ylabel("Density", fontweight="bold", fontsize=14)
    ax.set_title(title, fontweight="bold", fontsize=18)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 25)
    ax.tick_params(axis="both", which="major", labelsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, f"{filename}.png"), dpi=300, bbox_inches="tight")
    plt.close()


class CMeans:
    def __init__(self, n_clusters=3, m=2.0, error=0.005, maxiter=1000):
        self.n_clusters = n_clusters
        self.m = m
        self.error = error
        self.maxiter = maxiter
        self.labels_ = None
        self.u_ = None

    def fit_predict(self, X):
        X_reshaped = X.T.astype(np.float64)
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            X_reshaped, self.n_clusters, self.m, error=self.error, maxiter=self.maxiter
        )
        self.u_ = u
        self.labels_ = np.argmax(u, axis=0)
        return self.labels_

    def fit_transform(self, X):
        X_reshaped = X.T.astype(np.float64)
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            X_reshaped, self.n_clusters, self.m, error=self.error, maxiter=self.maxiter
        )
        self.u_ = u
        self.labels_ = np.argmax(u, axis=0)
        return u.T


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
        [Exponential(loc=2.25, rate=1.0), Exponential(loc=2.5, rate=0.5), Exponential(loc=2.75, rate=0.6667)],
        [1 / 3, 1 / 3, 1 / 3],
    )


def run_kmeans_comparison(n_samples, experiment_num=0):
    print(f"Running KMeans comparison for {n_samples} samples, experiment {experiment_num + 1}/{N_EXPERIMENTS}...")
    initial_mixture = create_random_initial_mixture()
    true_mixture = create_true_mixture()
    X = true_mixture.generate(n_samples)

    methods = {
        "kmeans_fast": ClusterizeInitializer(
            is_accurate=False, is_soft=False, clusterizer=KMeans(n_clusters=3, n_init=10)
        ),
        "kmeans_accurate": ClusterizeInitializer(
            is_accurate=True, is_soft=False, clusterizer=KMeans(n_clusters=3, n_init=10)
        ),
    }

    results = {}
    metrics_data = []

    true_labels = np.zeros(len(X))
    true_labels[len(X) // 3 : 2 * len(X) // 3] = 1
    true_labels[2 * len(X) // 3 :] = 2
    silhouette, db = calculate_clustering_metrics(X, true_labels)

    initial_kl = calculate_kl_divergence(initial_mixture, true_mixture)

    metrics_data.append(
        {
            "n_samples": n_samples,
            "method": "true",
            "silhouette": silhouette,
            "db": db,
            "kl_divergence": np.nan,
            "time": np.nan,
            "experiment": experiment_num,
        }
    )

    metrics_data.append(
        {
            "n_samples": n_samples,
            "method": "initial",
            "silhouette": np.nan,
            "db": np.nan,
            "kl_divergence": initial_kl,
            "time": np.nan,
            "experiment": experiment_num,
        }
    )

    for method_name, initializer in methods.items():
        start_time = time.time()
        mixture = initializer.perform(
            X=X.reshape(-1, 1),
            dists=initial_mixture.components,
            cluster_match_strategy=ClusterMatchStrategy.AKAIKE,
            estimation_strategies=[
                EstimationStrategy.QFUNCTION,
                EstimationStrategy.QFUNCTION,
                EstimationStrategy.QFUNCTION,
            ],
        )
        exec_time = time.time() - start_time

        kl_div = calculate_kl_divergence(mixture, true_mixture)
        results[method_name] = mixture

        silhouette, db = calculate_clustering_metrics(X, initializer.clusterizer.labels_)
        metrics_data.append(
            {
                "n_samples": n_samples,
                "method": method_name,
                "silhouette": silhouette,
                "db": db,
                "kl_divergence": kl_div,
                "time": exec_time,
                "experiment": experiment_num,
            }
        )

    if experiment_num == 0:
        plot_comparison_kmeans(
            X,
            true_mixture,
            initial_mixture,
            results,
            f"KMeans Comparison - {n_samples} samples",
            f"kmeans_comparison_{n_samples}samples",
        )

    return metrics_data


def run_cmeans_comparison(n_samples, experiment_num=0):
    print(f"Running CMeans comparison for {n_samples} samples, experiment {experiment_num + 1}/{N_EXPERIMENTS}...")
    initial_mixture = create_random_initial_mixture()
    true_mixture = create_true_mixture()
    X = true_mixture.generate(n_samples)

    methods = {
        "cmeans_fast": ClusterizeInitializer(
            is_accurate=False,
            is_soft=True,
            clusterizer=CMeans(n_clusters=3, m=2.0, error=0.005, maxiter=1000),
        ),
        "cmeans_accurate": ClusterizeInitializer(
            is_accurate=True,
            is_soft=True,
            clusterizer=CMeans(n_clusters=3, m=2.0, error=0.005, maxiter=1000),
        ),
    }

    results = {}
    metrics_data = []

    true_labels = np.zeros(len(X))
    true_labels[len(X) // 3 : 2 * len(X) // 3] = 1
    true_labels[2 * len(X) // 3 :] = 2
    silhouette, db = calculate_clustering_metrics(X, true_labels)

    initial_kl = calculate_kl_divergence(initial_mixture, true_mixture)

    metrics_data.append(
        {
            "n_samples": n_samples,
            "method": "true",
            "silhouette": silhouette,
            "db": db,
            "kl_divergence": np.nan,
            "time": np.nan,
            "experiment": experiment_num,
        }
    )

    metrics_data.append(
        {
            "n_samples": n_samples,
            "method": "initial",
            "silhouette": np.nan,
            "db": np.nan,
            "kl_divergence": initial_kl,
            "time": np.nan,
            "experiment": experiment_num,
        }
    )

    for method_name, initializer in methods.items():
        start_time = time.time()
        mixture = initializer.perform(
            X=X.reshape(-1, 1),
            dists=initial_mixture.components,
            cluster_match_strategy=ClusterMatchStrategy.AKAIKE,
            estimation_strategies=[
                EstimationStrategy.QFUNCTION,
                EstimationStrategy.QFUNCTION,
                EstimationStrategy.QFUNCTION,
            ],
        )
        exec_time = time.time() - start_time

        kl_div = calculate_kl_divergence(mixture, true_mixture)
        results[method_name] = mixture

        silhouette, db = calculate_clustering_metrics(X, initializer.clusterizer.labels_)
        metrics_data.append(
            {
                "n_samples": n_samples,
                "method": method_name,
                "silhouette": silhouette,
                "db": db,
                "kl_divergence": kl_div,
                "time": exec_time,
                "experiment": experiment_num,
            }
        )

    if experiment_num == 0:
        plot_comparison_cmeans(
            X,
            true_mixture,
            initial_mixture,
            results,
            f"CMeans Comparison - {n_samples} samples",
            f"cmeans_comparison_{n_samples}samples",
        )

    return metrics_data


sample_sizes = [200, 500, 1000]
all_metrics = []

np.random.seed(35)

for n_samples in sample_sizes:
    for exp_num in range(N_EXPERIMENTS):
        kmeans_metrics = run_kmeans_comparison(n_samples, exp_num)
        cmeans_metrics = run_cmeans_comparison(n_samples, exp_num)
        all_metrics.extend(kmeans_metrics)
        all_metrics.extend(cmeans_metrics)

results_df = pd.DataFrame(all_metrics)

agg_results = (
    results_df.groupby(["n_samples", "method"])
    .agg(
        {
            "silhouette": ["mean", "std"],
            "db": ["mean", "std"],
            "kl_divergence": ["mean", "std"],
            "time": ["mean", "std"],
        }
    )
    .round(4)
)

formatted_results = []
for (n_samples, method), row in agg_results.iterrows():
    formatted_row = {
        "n_samples": n_samples,
        "method": method,
        "silhouette": format_with_std(row[("silhouette", "mean")], row[("silhouette", "std")]),
        "db": format_with_std(row[("db", "mean")], row[("db", "std")]),
        "kl_divergence": format_with_std(row[("kl_divergence", "mean")], row[("kl_divergence", "std")]),
        "time": format_with_std(row[("time", "mean")], row[("time", "std")]),
    }
    formatted_results.append(formatted_row)

formatted_df = pd.DataFrame(formatted_results)

print("\n" + "=" * 100)
print("CLUSTERING METRICS RESULTS (with standard deviation)")
print("=" * 100)
print(formatted_df.to_string(index=False))

results_df.to_csv(os.path.join(results_path, "clustering_metrics_results_detailed.csv"), index=False)
formatted_df.to_csv(os.path.join(results_path, "clustering_metrics_results.csv"), index=False)
print("\nDetailed results saved to results/clustering_metrics_results_detailed.csv")
print("Formatted results saved to results/clustering_metrics_results.csv")

save_table_as_image(formatted_df, "clustering_metrics_results", "Clustering Metrics Results (Mean ± Std)")

kmeans_df = results_df[results_df["method"].str.contains("kmeans|true|initial")]
cmeans_df = results_df[results_df["method"].str.contains("cmeans|true|initial")]


def create_aggregated_summary(df):
    summary_data = []
    for n_samples in sample_sizes:
        for method in df["method"].unique():
            method_data = df[(df["n_samples"] == n_samples) & (df["method"] == method)]
            if len(method_data) > 0:
                row = {
                    "n_samples": n_samples,
                    "method": method,
                    "silhouette": format_with_std(method_data["silhouette"].mean(), method_data["silhouette"].std()),
                    "db": format_with_std(method_data["db"].mean(), method_data["db"].std()),
                    "kl_divergence": format_with_std(
                        method_data["kl_divergence"].mean(), method_data["kl_divergence"].std()
                    ),
                    "time": format_with_std(method_data["time"].mean(), method_data["time"].std()),
                }
                summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    pivot_df = summary_df.pivot_table(
        index="method", columns="n_samples", values=["silhouette", "db", "kl_divergence", "time"], aggfunc="first"
    )
    return pivot_df


kmeans_summary = create_aggregated_summary(kmeans_df)
print("\nKMEANS SUMMARY TABLE (with standard deviation)")
print("=" * 80)
print(kmeans_summary)
kmeans_summary.to_csv(os.path.join(results_path, "kmeans_summary_table.csv"))
save_multiindex_table_as_image(kmeans_summary, "kmeans_summary_table", "KMeans Summary (Mean ± Std)")

cmeans_summary = create_aggregated_summary(cmeans_df)
print("\nCMEANS SUMMARY TABLE (with standard deviation)")
print("=" * 80)
print(cmeans_summary)
cmeans_summary.to_csv(os.path.join(results_path, "cmeans_summary_table.csv"))
save_multiindex_table_as_image(cmeans_summary, "cmeans_summary_table", "CMeans Summary (Mean ± Std)")

combined_summary = create_aggregated_summary(results_df)
print("\nCOMBINED SUMMARY TABLE (with standard deviation)")
print("=" * 100)
print(combined_summary)
combined_summary.to_csv(os.path.join(results_path, "combined_summary_table.csv"))
save_multiindex_table_as_image(combined_summary, "combined_summary_table", "Combined Summary (Mean ± Std)")

kl_agg = results_df.groupby(["method", "n_samples"])["kl_divergence"].agg(["mean", "std"]).round(4)
kl_summary_data = []
for (method, n_samples), row in kl_agg.iterrows():
    kl_summary_data.append(
        {"method": method, "n_samples": n_samples, "kl_divergence": format_with_std(row["mean"], row["std"])}
    )
kl_summary_df = pd.DataFrame(kl_summary_data)
kl_summary = kl_summary_df.pivot_table(index="method", columns="n_samples", values="kl_divergence", aggfunc="first")

print("\nKL DIVERGENCE SUMMARY (with standard deviation)")
print("=" * 100)
print(kl_summary)
kl_summary.to_csv(os.path.join(results_path, "kl_summary.csv"))
save_table_as_image(kl_summary, "kl_summary", "KL Divergence Summary (Mean ± Std)")

time_data = results_df[~results_df["method"].isin(["true", "initial"])]
time_agg = time_data.groupby(["method", "n_samples"])["time"].agg(["mean", "std"]).round(4)
time_summary_data = []
for (method, n_samples), row in time_agg.iterrows():
    time_summary_data.append(
        {"method": method, "n_samples": n_samples, "time": format_with_std(row["mean"], row["std"])}
    )
time_summary_df = pd.DataFrame(time_summary_data)
time_summary = time_summary_df.pivot_table(index="method", columns="n_samples", values="time", aggfunc="first")

print("\nTIME SUMMARY (seconds) (with standard deviation)")
print("=" * 100)
print(time_summary)
time_summary.to_csv(os.path.join(results_path, "time_summary.csv"))
save_table_as_image(time_summary, "time_summary", "Time Summary (Mean ± Std)")

valid_results = results_df[~results_df["method"].isin(["true", "initial"])]
best_methods_list = []

for n_size in sample_sizes:
    n_data = valid_results[valid_results["n_samples"] == n_size]

    for exp_num in range(N_EXPERIMENTS):
        exp_data = n_data[n_data["experiment"] == exp_num]

        if len(exp_data) > 0:
            best_silhouette_idx = exp_data["silhouette"].idxmax()
            best_silhouette_row = exp_data.loc[best_silhouette_idx]

            best_db_idx = exp_data["db"].idxmin()
            best_db_row = exp_data.loc[best_db_idx]

            best_kl_idx = exp_data["kl_divergence"].idxmin()
            best_kl_row = exp_data.loc[best_kl_idx]

            best_methods_list.extend(
                [
                    {
                        "n_samples": n_size,
                        "metric": "best_silhouette",
                        "method": best_silhouette_row["method"],
                        "silhouette": best_silhouette_row["silhouette"],
                        "db": best_silhouette_row["db"],
                        "kl_divergence": best_silhouette_row["kl_divergence"],
                        "time": best_silhouette_row["time"],
                        "experiment": exp_num,
                    },
                    {
                        "n_samples": n_size,
                        "metric": "best_db",
                        "method": best_db_row["method"],
                        "silhouette": best_db_row["silhouette"],
                        "db": best_db_row["db"],
                        "kl_divergence": best_db_row["kl_divergence"],
                        "time": best_db_row["time"],
                        "experiment": exp_num,
                    },
                    {
                        "n_samples": n_size,
                        "metric": "best_kl",
                        "method": best_kl_row["method"],
                        "silhouette": best_kl_row["silhouette"],
                        "db": best_kl_row["db"],
                        "kl_divergence": best_kl_row["kl_divergence"],
                        "time": best_kl_row["time"],
                        "experiment": exp_num,
                    },
                ]
            )

best_methods_detailed = pd.DataFrame(best_methods_list)

best_methods_agg = (
    best_methods_detailed.groupby(["n_samples", "metric", "method"])
    .agg(
        {
            "silhouette": ["mean", "std"],
            "db": ["mean", "std"],
            "kl_divergence": ["mean", "std"],
            "time": ["mean", "std"],
        }
    )
    .round(4)
)

best_methods_final = []
for (n_samples, metric, method), row in best_methods_agg.iterrows():
    best_methods_final.append(
        {
            "n_samples": n_samples,
            "metric": metric,
            "method": method,
            "silhouette": format_with_std(row[("silhouette", "mean")], row[("silhouette", "std")]),
            "db": format_with_std(row[("db", "mean")], row[("db", "std")]),
            "kl_divergence": format_with_std(row[("kl_divergence", "mean")], row[("kl_divergence", "std")]),
            "time": format_with_std(row[("time", "mean")], row[("time", "std")]),
        }
    )

best_methods_final_df = pd.DataFrame(best_methods_final)
print("\nBEST METHODS BY SAMPLE SIZE (with standard deviation)")
print("=" * 100)
print(best_methods_final_df.to_string(index=False))
best_methods_detailed.to_csv(os.path.join(results_path, "best_methods_detailed.csv"), index=False)
best_methods_final_df.to_csv(os.path.join(results_path, "best_methods.csv"), index=False)
save_table_as_image(best_methods_final_df, "best_methods", "Best Methods (Mean ± Std)")

n_500_data = results_df[results_df["n_samples"] == SAMPLES_ALL_COMPARISON].copy()
n_500_agg = (
    n_500_data.groupby("method")
    .agg(
        {
            "silhouette": ["mean", "std"],
            "db": ["mean", "std"],
            "kl_divergence": ["mean", "std"],
            "time": ["mean", "std"],
        }
    )
    .round(4)
)

n_500_summary_data = []
for method, row in n_500_agg.iterrows():
    n_500_summary_data.append(
        {
            "method": method,
            "silhouette": format_with_std(row[("silhouette", "mean")], row[("silhouette", "std")]),
            "db": format_with_std(row[("db", "mean")], row[("db", "std")]),
            "kl_divergence": format_with_std(row[("kl_divergence", "mean")], row[("kl_divergence", "std")]),
            "time": format_with_std(row[("time", "mean")], row[("time", "std")]),
        }
    )

n_500_summary = pd.DataFrame(n_500_summary_data).set_index("method")

print("\n" + "=" * 100)
print("COMBINED SUMMARY FOR 500 SAMPLES (with standard deviation)")
print("=" * 100)
print(n_500_summary)

n_500_summary.to_csv(os.path.join(results_path, "combined_500_samples.csv"))
save_presentation_table(
    n_500_summary, "combined_500_samples_presentation", "Combined Results for 500 Samples (Mean ± Std)"
)

print("\nExperiment completed!")
print(f"Each sample size was tested {N_EXPERIMENTS} times")
print("Generated files:")
print("- clustering_metrics_results_detailed.csv (detailed results for all experiments)")
print("- clustering_metrics_results.csv/.png (formatted results with mean ± std)")
print("- kmeans_summary_table.csv/.png")
print("- cmeans_summary_table.csv/.png")
print("- combined_summary_table.csv/.png")
print("- combined_500_samples.csv/.png")
print("- combined_500_samples_presentation.png (for presentation)")
print("- kl_summary.csv/.png")
print("- time_summary.csv/.png")
print("- best_methods_detailed.csv (detailed best methods)")
print("- best_methods.csv/.png (aggregated best methods)")
print("- Comparison plots for KMeans and CMeans (for first experiment only)")
