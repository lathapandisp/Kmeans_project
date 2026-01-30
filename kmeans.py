import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


np.random.seed(42)

X, y_true = make_blobs(
    n_samples=300,
    centers=5,
    cluster_std=1.2,
    random_state=42
)

plt.figure()
plt.scatter(X[:, 0], X[:, 1])
plt.title("Synthetic Dataset (300 samples, 5 clusters)")
plt.show()



class KMeansScratch:
    def __init__(self, k=3, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol

    def initialize_centroids(self, X):
        indices = np.random.choice(X.shape[0], self.k, replace=False)
        return X[indices]

    def compute_distances(self, X, centroids):
        # Euclidean distance
        return np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

    def assign_clusters(self, distances):
        return np.argmin(distances, axis=1)

    def update_centroids(self, X, labels):
        centroids = []
        for i in range(self.k):
            points = X[labels == i]
            if len(points) == 0:
                # Handle empty cluster
                centroids.append(X[np.random.randint(0, X.shape[0])])
            else:
                centroids.append(points.mean(axis=0))
        return np.array(centroids)

    def fit(self, X):
        self.centroids = self.initialize_centroids(X)

        for iteration in range(self.max_iters):
            distances = self.compute_distances(X, self.centroids)
            self.labels = self.assign_clusters(distances)
            new_centroids = self.update_centroids(X, self.labels)

            shift = np.linalg.norm(new_centroids - self.centroids)

            if shift < self.tol:
                print(f"Converged in {iteration+1} iterations")
                break

            self.centroids = new_centroids

        return self.labels

    def compute_wcss(self, X):
        distances = self.compute_distances(X, self.centroids)
        min_distances = np.min(distances, axis=1)
        return np.sum(min_distances ** 2)



wcss = []
k_values = range(1, 11)

for k in k_values:
    model = KMeansScratch(k=k)
    model.fit(X)
    wcss.append(model.compute_wcss(X))

plt.figure()
plt.plot(k_values, wcss, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

print("\nWCSS Values:")
for k, value in zip(k_values, wcss):
    print(f"K = {k}, WCSS = {value:.2f}")



optimal_k = 5   

final_model = KMeansScratch(k=optimal_k)
labels = final_model.fit(X)
centroids = final_model.centroids

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='X', s=200)
plt.title(f"Final Clustering (K = {optimal_k})")
plt.show()


print("\n--- Interpretation ---")
print("The Elbow plot shows a sharp decrease in WCSS up to K=5.")
print("After K=5, the reduction becomes gradual, indicating diminishing returns.")
print("Thus, K=5 is selected as the optimal number of clusters.")
print("The final clustering aligns well with the synthetic data structure.")
