import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import skfuzzy as fuzz

from rework_pysatl_mpest import Exponential, MixtureModel
from rework_pysatl_mpest.Initializers.clusterizeInitializer import ClusterizeInitializer
from rework_pysatl_mpest.Initializers.strategies import ClusterMatchStrategy, EstimationStrategy

os.makedirs("results", exist_ok=True)

plt.style.use("default")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = [12, 8]
plt.rcParams["font.size"] = 12
plt.rcParams["font.weight"] = "bold"


def plot_comparison(original_params, new_params, X, title, ax1, ax2):
    original_mixture = MixtureModel(
        [
            Exponential(loc=original_params[0][0], rate=original_params[0][1]),
            Exponential(loc=original_params[1][0], rate=original_params[1][1]),
            Exponential(loc=original_params[2][0], rate=original_params[2][1]),
        ],
        [1 / 3, 1 / 3, 1 / 3],
    )

    new_mixture = MixtureModel(
        [
            Exponential(loc=new_params[0][0], rate=new_params[0][1]),
            Exponential(loc=new_params[1][0], rate=new_params[1][1]),
            Exponential(loc=new_params[2][0], rate=new_params[2][1]),
        ],
        [1 / 3, 1 / 3, 1 / 3],
    )

    ax1.hist(X, bins=50, density=True, alpha=0.7, color="lightblue", edgecolor="black", label="Data")
    x_plot = np.linspace(0, 25, 1000)
    ax1.plot(x_plot, [original_mixture.pdf(xi) for xi in x_plot], "r-", linewidth=3, label="Initial mixture", alpha=0.8)
    ax1.plot(x_plot, [new_mixture.pdf(xi) for xi in x_plot], "g--", linewidth=3, label="New mixture", alpha=0.8)
    ax1.set_xlabel("Value")
    ax1.set_ylabel("Density")
    ax1.set_title(f"{title} - Density distributions")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 25)

    ax2.hist(X, bins=50, density=True, alpha=0.7, color="lightblue", edgecolor="black", label="Data")
    ax2.plot(x_plot, [original_mixture.pdf(xi) for xi in x_plot], "r-", linewidth=3, label="Initial mixture", alpha=0.8)
    ax2.plot(x_plot, [new_mixture.pdf(xi) for xi in x_plot], "g--", linewidth=3, label="New mixture", alpha=0.8)
    ax2.set_yscale("log")
    ax2.set_xlabel("Value")
    ax2.set_ylabel("Log density")
    ax2.set_title(f"{title} - Logarithmic scale")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 25)
    ax2.set_ylim(1e-4, 1)


class FuzzyCMeansClusterizer:
    def __init__(self, n_clusters=3, m=2.0, error=0.005, maxiter=1000):
        self.n_clusters = n_clusters
        self.m = m
        self.error = error
        self.maxiter = maxiter

    def fit_predict(self, X):
        X_reshaped = X.T.astype(np.float64)
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            X_reshaped, self.n_clusters, self.m, error=self.error, maxiter=self.maxiter
        )
        return np.argmax(u, axis=0)

    def fit_transform(self, X):
        X_reshaped = X.T.astype(np.float64)
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            X_reshaped, self.n_clusters, self.m, error=self.error, maxiter=self.maxiter
        )
        return u.T


np.random.seed(42)
n_samples = 450
data1 = np.random.exponential(scale=1.0, size=n_samples // 3) + 2.0
data2 = np.random.exponential(scale=2.0, size=n_samples // 3) + 5.0
data3 = np.random.exponential(scale=1.5, size=n_samples // 3) + 8.0
X = np.concatenate([data1, data2, data3])
np.random.shuffle(X)

initial_exp1 = Exponential(loc=0.0, rate=0.1)
initial_exp2 = Exponential(loc=5.0, rate=0.05)
initial_exp3 = Exponential(loc=10.0, rate=0.01)

fuzzy_cmeans = FuzzyCMeansClusterizer(n_clusters=3, m=2.0, error=0.005, maxiter=1000)
initializer = ClusterizeInitializer(is_accurate=True, is_soft=True, clusterizer=fuzzy_cmeans)

accurate_mixture = None
fast_mixture = None

try:
    accurate_mixture = initializer.perform(
        X=X.reshape(-1, 1),
        dists=[Exponential(loc=0.0, rate=0.1), Exponential(loc=5.0, rate=0.05), Exponential(loc=10.0, rate=0.01)],
        cluster_match_info=ClusterMatchStrategy.AKAIKE,
        estimation_info=[EstimationStrategy.QFUNCTION, EstimationStrategy.QFUNCTION, EstimationStrategy.QFUNCTION],
    )
except Exception as e:
    print(f"Fuzzy C-Means initialization error: {e}")

fast_initializer = ClusterizeInitializer(is_accurate=False, is_soft=True, clusterizer=fuzzy_cmeans)

try:
    fast_mixture = fast_initializer.perform(
        X=X.reshape(-1, 1),
        dists=[Exponential(loc=0.0, rate=0.1), Exponential(loc=5.0, rate=0.05), Exponential(loc=10.0, rate=0.01)],
        cluster_match_info=ClusterMatchStrategy.AKAIKE,
        estimation_info=[EstimationStrategy.QFUNCTION, EstimationStrategy.QFUNCTION, EstimationStrategy.QFUNCTION],
    )
except Exception as e:
    print(f"Fast Fuzzy C-Means initialization error: {e}")

hard_initializer = ClusterizeInitializer(is_accurate=True, is_soft=False, clusterizer=fuzzy_cmeans)
hard_mixture = None

try:
    hard_mixture = hard_initializer.perform(
        X=X.reshape(-1, 1),
        dists=[Exponential(loc=0.0, rate=0.1), Exponential(loc=5.0, rate=0.05), Exponential(loc=10.0, rate=0.01)],
        cluster_match_info=ClusterMatchStrategy.AKAIKE,
        estimation_info=[EstimationStrategy.QFUNCTION, EstimationStrategy.QFUNCTION, EstimationStrategy.QFUNCTION],
    )
except Exception as e:
    print(f"Hard Fuzzy C-Means initialization error: {e}")

original_params = [(0.0, 0.1), (5.0, 0.05), (10.0, 0.01)]

if accurate_mixture is not None:
    accurate_params = [
        (accurate_mixture.components[0].loc, accurate_mixture.components[0].rate),
        (accurate_mixture.components[1].loc, accurate_mixture.components[1].rate),
        (accurate_mixture.components[2].loc, accurate_mixture.components[2].rate),
    ]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    plot_comparison(original_params, accurate_params, X, "Fuzzy C-Means (soft) initialization", ax1, ax2)
    plt.tight_layout()
    plt.savefig("results/fuzzy_cmeans_soft_initialization_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

if fast_mixture is not None:
    fast_params = [
        (fast_mixture.components[0].loc, fast_mixture.components[0].rate),
        (fast_mixture.components[1].loc, fast_mixture.components[1].rate),
        (fast_mixture.components[2].loc, fast_mixture.components[2].rate),
    ]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    plot_comparison(original_params, fast_params, X, "Fuzzy C-Means (fast) initialization", ax1, ax2)
    plt.tight_layout()
    plt.savefig("results/fuzzy_cmeans_fast_initialization_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

if hard_mixture is not None:
    hard_params = [
        (hard_mixture.components[0].loc, hard_mixture.components[0].rate),
        (hard_mixture.components[1].loc, hard_mixture.components[1].rate),
        (hard_mixture.components[2].loc, hard_mixture.components[2].rate),
    ]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    plot_comparison(original_params, hard_params, X, "Fuzzy C-Means (hard) initialization", ax1, ax2)
    plt.tight_layout()
    plt.savefig("results/fuzzy_cmeans_hard_initialization_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

if accurate_mixture is not None and fast_mixture is not None and hard_mixture is not None:
    fig, ax = plt.subplots(figsize=(14, 8))
    x_plot = np.linspace(0, 25, 1000)

    original_mixture = MixtureModel(
        [Exponential(loc=0.0, rate=0.1), Exponential(loc=5.0, rate=0.05), Exponential(loc=10.0, rate=0.01)],
        [1 / 3, 1 / 3, 1 / 3],
    )

    accurate_mixture_plot = MixtureModel(
        [
            Exponential(loc=accurate_mixture.components[0].loc, rate=accurate_mixture.components[0].rate),
            Exponential(loc=accurate_mixture.components[1].loc, rate=accurate_mixture.components[1].rate),
            Exponential(loc=accurate_mixture.components[2].loc, rate=accurate_mixture.components[2].rate),
        ],
        accurate_mixture.weights,
    )

    fast_mixture_plot = MixtureModel(
        [
            Exponential(loc=fast_mixture.components[0].loc, rate=fast_mixture.components[0].rate),
            Exponential(loc=fast_mixture.components[1].loc, rate=fast_mixture.components[1].rate),
            Exponential(loc=fast_mixture.components[2].loc, rate=fast_mixture.components[2].rate),
        ],
        fast_mixture.weights,
    )

    hard_mixture_plot = MixtureModel(
        [
            Exponential(loc=hard_mixture.components[0].loc, rate=hard_mixture.components[0].rate),
            Exponential(loc=hard_mixture.components[1].loc, rate=hard_mixture.components[1].rate),
            Exponential(loc=hard_mixture.components[2].loc, rate=hard_mixture.components[2].rate),
        ],
        hard_mixture.weights,
    )

    ax.hist(X, bins=50, density=True, alpha=0.3, color="gray", edgecolor="black", label="Data")
    ax.plot(x_plot, [original_mixture.pdf(xi) for xi in x_plot], "r-", linewidth=3, label="Initial mixture", alpha=0.8)
    ax.plot(
        x_plot,
        [accurate_mixture_plot.pdf(xi) for xi in x_plot],
        "g--",
        linewidth=3,
        label="Fuzzy C-Means (soft, accurate)",
        alpha=0.8,
    )
    ax.plot(
        x_plot,
        [fast_mixture_plot.pdf(xi) for xi in x_plot],
        "b:",
        linewidth=3,
        label="Fuzzy C-Means (soft, fast)",
        alpha=0.8,
    )
    ax.plot(
        x_plot,
        [hard_mixture_plot.pdf(xi) for xi in x_plot],
        "m-.",
        linewidth=3,
        label="Fuzzy C-Means (hard)",
        alpha=0.8,
    )

    ax.set_xlabel("Value", fontweight="bold")
    ax.set_ylabel("Density", fontweight="bold")
    ax.set_title("Comparison of Fuzzy C-Means initialization methods (3 components)", fontweight="bold", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 25)

    plt.tight_layout()
    plt.savefig("results/fuzzy_cmeans_all_methods_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
