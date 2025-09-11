import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.cluster import KMeans
from rework_pysatl_mpest import Exponential, MixtureModel
from rework_pysatl_mpest.estimators.iterative.Initializers.clusterizeInitializer import ClusterizeInitializer

os.makedirs("results", exist_ok=True)

plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'bold'


def plot_comparison(original_params, new_params, X, title, ax1, ax2):
    original_mixture = MixtureModel(
        [Exponential(loc=original_params[0][0], rate=original_params[0][1]),
         Exponential(loc=original_params[1][0], rate=original_params[1][1])],
        [0.5, 0.5]
    )

    new_mixture = MixtureModel(
        [Exponential(loc=new_params[0][0], rate=new_params[0][1]),
         Exponential(loc=new_params[1][0], rate=new_params[1][1])],
        [0.5, 0.5]
    )

    ax1.hist(X, bins=50, density=True, alpha=0.7, color='lightblue', edgecolor='black', label='Data')

    x_plot = np.linspace(0, 20, 1000)

    ax1.plot(x_plot, [original_mixture.pdf(xi) for xi in x_plot],
             'r-', linewidth=3, label='Initial mixture', alpha=0.8)

    ax1.plot(x_plot, [new_mixture.pdf(xi) for xi in x_plot],
             'g--', linewidth=3, label='New mixture', alpha=0.8)

    ax1.set_xlabel('Value')
    ax1.set_ylabel('Density')
    ax1.set_title(f'{title} - Density distributions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 20)

    ax2.hist(X, bins=50, density=True, alpha=0.7, color='lightblue', edgecolor='black', label='Data')
    ax2.plot(x_plot, [original_mixture.pdf(xi) for xi in x_plot],
             'r-', linewidth=3, label='Initial mixture', alpha=0.8)
    ax2.plot(x_plot, [new_mixture.pdf(xi) for xi in x_plot],
             'g--', linewidth=3, label='New mixture', alpha=0.8)

    ax2.set_yscale('log')
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Log density')
    ax2.set_title(f'{title} - Logarithmic scale')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 20)
    ax2.set_ylim(1e-4, 1)


np.random.seed(42)
n_samples = 300

data1 = np.random.exponential(scale=1.0, size=n_samples // 2) + 2.0
data2 = np.random.exponential(scale=2.0, size=n_samples // 2) + 5.0
X = np.concatenate([data1, data2])
np.random.shuffle(X)

print("=" * 60)
print("Test ClusterizeInitializer")
print("=" * 60)

initial_exp1 = Exponential(loc=0.0, rate=0.1)
initial_exp2 = Exponential(loc=10.0, rate=0.01)

print(initial_exp1.__class__)

print("Initial distributions:")
print(f"1. {initial_exp1}")
print(f"2. {initial_exp2}")

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)

initializer = ClusterizeInitializer(
    is_accurate=True,
    is_soft=False,
    clusterizer=kmeans
)

accurate_mixture = None
fast_mixture = None

try:
    print("\nRunning accurate initialization...")
    accurate_mixture = initializer.perform(
        x=X.reshape(-1, 1),
        dists=[Exponential(loc=0.0, rate=0.1), Exponential(loc=10.0, rate=0.01)],
        info=[lambda: None, lambda: None]
    )

    print("Initialization successful!")
    print(f"Number of components: {len(accurate_mixture.components)}")
    print(f"Weights: {accurate_mixture.weights}")

    for i, dist in enumerate(accurate_mixture.components):
        print(f"\nComponent {i + 1}:")
        print(f"  Type: {dist.name}")
        print(f"  Parameters: loc={dist.loc:.3f}, rate={dist.rate:.3f}")
        print(f"  Parameter vector: {dist.get_params_vector(list(dist.params))}")

    print(f"\nParameters were recalculated:")
    print(f"  Was: loc=0.0, rate=0.1 -> Now: loc={accurate_mixture.components[0].loc:.3f}, rate={accurate_mixture.components[0].rate:.3f}")
    print(f"  Was: loc=10.0, rate=0.01 -> Now: loc={accurate_mixture.components[1].loc:.3f}, rate={accurate_mixture.components[1].rate:.3f}")

except Exception as e:
    print(f"Initialization error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 50)
print("Testing fast initialization:")

fast_initializer = ClusterizeInitializer(
    is_accurate=False,
    is_soft=False,
    clusterizer=KMeans(n_clusters=2, random_state=42, n_init=10)
)

try:
    print("Running fast initialization...")
    fast_mixture = fast_initializer.perform(
        x=X.reshape(-1, 1),
        dists=[Exponential(loc=0.0, rate=0.1), Exponential(loc=10.0, rate=0.01)],
        info=[lambda: None, lambda: None]
    )

    print("Fast initialization successful!")
    print(f"Weights: {fast_mixture.weights}")

    for i, dist in enumerate(fast_mixture.components):
        print(f"Component {i + 1}: {dist.name} with parameters loc={dist.loc:.3f}, rate={dist.rate:.3f}")

    print(f"\nParameters were recalculated:")
    print(f"  Was: loc=0.0, rate=0.1 -> Now: loc={fast_mixture.components[0].loc:.3f}, rate={fast_mixture.components[0].rate:.3f}")
    print(f"  Was: loc=10.0, rate=0.01 -> Now: loc={fast_mixture.components[1].loc:.3f}, rate={fast_mixture.components[1].rate:.3f}")

except Exception as e:
    print(f"Fast initialization error: {e}")

original_params = [(0.0, 0.1), (10.0, 0.01)]

if accurate_mixture is not None:
    accurate_params = [
        (accurate_mixture.components[0].loc, accurate_mixture.components[0].rate),
        (accurate_mixture.components[1].loc, accurate_mixture.components[1].rate)
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    plot_comparison(original_params, accurate_params, X, "Accurate initialization", ax1, ax2)
    plt.tight_layout()
    plt.savefig('results/accurate_initialization_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Accurate initialization plot saved to results/accurate_initialization_comparison.png")

if fast_mixture is not None:
    fast_params = [
        (fast_mixture.components[0].loc, fast_mixture.components[0].rate),
        (fast_mixture.components[1].loc, fast_mixture.components[1].rate)
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    plot_comparison(original_params, fast_params, X, "Fast initialization", ax1, ax2)
    plt.tight_layout()
    plt.savefig('results/fast_initialization_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Fast initialization plot saved to results/fast_initialization_comparison.png")

if accurate_mixture is not None and fast_mixture is not None:
    fig, ax = plt.subplots(figsize=(14, 8))

    x_plot = np.linspace(0, 20, 1000)

    original_mixture = MixtureModel(
        [Exponential(loc=0.0, rate=0.1), Exponential(loc=10.0, rate=0.01)],
        [0.5, 0.5]
    )

    accurate_mixture_plot = MixtureModel(
        [Exponential(loc=accurate_mixture.components[0].loc, rate=accurate_mixture.components[0].rate),
         Exponential(loc=accurate_mixture.components[1].loc, rate=accurate_mixture.components[1].rate)],
        [0.5, 0.5]
    )

    fast_mixture_plot = MixtureModel(
        [Exponential(loc=fast_mixture.components[0].loc, rate=fast_mixture.components[0].rate),
         Exponential(loc=fast_mixture.components[1].loc, rate=fast_mixture.components[1].rate)],
        [0.5, 0.5]
    )

    ax.hist(X, bins=50, density=True, alpha=0.3, color='gray', edgecolor='black', label='Data')

    ax.plot(x_plot, [original_mixture.pdf(xi) for xi in x_plot],
            'r-', linewidth=3, label='Initial mixture', alpha=0.8)
    ax.plot(x_plot, [accurate_mixture_plot.pdf(xi) for xi in x_plot],
            'g--', linewidth=3, label='Accurate initialization', alpha=0.8)
    ax.plot(x_plot, [fast_mixture_plot.pdf(xi) for xi in x_plot],
            'b:', linewidth=3, label='Fast initialization', alpha=0.8)

    ax.set_xlabel('Value', fontweight='bold')
    ax.set_ylabel('Density', fontweight='bold')
    ax.set_title('Comparison of all initialization methods', fontweight='bold', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 20)

    plt.tight_layout()
    plt.savefig('results/all_methods_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("All methods comparison plot saved to results/all_methods_comparison.png")

print("\n" + "=" * 60)
print("TESTING COMPLETED!")
print("All plots saved to 'results' folder")
print("=" * 60)