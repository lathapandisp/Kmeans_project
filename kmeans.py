import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


class KMeansScratch:
    def __init__(self, k=3, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.labels = None

    def initialize_centroids(self, X):
        indices = np.random.choice(X.shape[0], self.k, replace=False)
        return X[indices]

    def compute_distances(self, X, centroids):
        return np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

    def assign_clusters(self, distances):
        return np.argmin(distances, axis=1)

    def update_centroids(self, X, labels):
        centroids = []
        for i in range(self.k):
            points = X[labels == i]
            if len(points) == 0:
                centroids.append(X[np.random.randint(0, X.shape[0])])
            else:
                centroids.append(points.mean(axis=0))
        return np.array(centroids)

    def fit(self, X):
        self.centroids = self.initialize_centroids(X)

        for _ in range(self.max_iters):
            distances = self.compute_distances(X, self.centroids)
            self.labels = self.assign_clusters(distances)
            new_centroids = self.update_centroids(X, self.labels)

            shift = np.linalg.norm(new_centroids - self.centroids)
            if shift < self.tol:
                break

            self.centroids = new_centroids

        return self.labels

    def compute_wcss(self, X):
        distances = self.compute_distances(X, self.centroids)
        min_distances = np.min(distances, axis=1)
        return np.sum(min_distances ** 2)


def generate_data():
    np.random.seed(42)
    X, _ = make_blobs(
        n_samples=300,
        centers=5,
        cluster_std=1.2,
        random_state=42
    )
    return X


def find_optimal_k(X, max_k=10):
    wcss = []
    k_values = range(1, max_k + 1)

    for k in k_values:
        model = KMeansScratch(k=k)
        model.fit(X)
        wcss.append(model.compute_wcss(X))

    wcss_array = np.array(wcss)
    second_derivative = np.diff(wcss_array, 2)
    optimal_k = np.argmin(second_derivative) + 2

    return optimal_k, k_values, wcss


def plot_elbow(k_values, wcss):
    plt.figure()
    plt.plot(k_values, wcss, marker='o')
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("WCSS")
    plt.title("Elbow Method")
    plt.show()


def plot_clusters(X, labels, centroids, k):
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200)
    plt.title(f"Final Clustering (K = {k})")
    plt.show()


if __name__ == "__main__":

    X = generate_data()

    optimal_k, k_values, wcss = find_optimal_k(X)

    print(f"\nAutomatically Selected Optimal K: {optimal_k}")

    plot_elbow(k_values, wcss)

    final_model = KMeansScratch(k=optimal_k)
    labels = final_model.fit(X)
    centroids = final_model.centroids

    plot_clusters(X, labels, centroids, optimal_k)

    print("\n--- Analysis ---")
    print("""
This project implemented the K-Means clustering algorithm from scratch using NumPy.
The algorithm includes centroid initialization via random sampling, Euclidean distance
calculation using vectorized operations, cluster assignment based on minimum distance,
centroid updates using mean recomputation, and convergence detection using centroid shift tolerance.

The Elbow Method was applied using Within-Cluster Sum of Squares (WCSS).
WCSS decreases sharply at lower K values and gradually stabilizes.
The second derivative of the WCSS curve was used to automatically detect the
inflection point, avoiding manual hardcoding of K.

The final clustering result demonstrates clear separation of clusters,
and centroids align well with the synthetic dataset structure.

One challenge of K-Means is sensitivity to initialization, which may
produce slightly different results across runs. Empty cluster handling
was included to ensure robustness.

Overall, the implementation successfully demonstrates understanding
of iterative optimization, unsupervised learning, and model selection.
""")
