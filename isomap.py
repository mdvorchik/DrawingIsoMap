import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svd
from scipy.sparse.csgraph import shortest_path
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import Isomap

X, color = make_swiss_roll(n_samples=1000, random_state=2022)

euclidean_distances_3d = np.zeros((1000, 1000))
for i in range(len(X)):
    for j in range(len(X)):
        dist = np.linalg.norm(X[i] - X[j])
        euclidean_distances_3d[i, j] = dist

distances_3d_with_only_neighbours = np.zeros((1000, 1000))
for i in range(len(euclidean_distances_3d)):
    limit_min_dist_neighbour = sorted(euclidean_distances_3d[:, i])[10]
    for j in range(len(euclidean_distances_3d)):
        dist = euclidean_distances_3d[i, j] if euclidean_distances_3d[i, j] < limit_min_dist_neighbour else 1000_000_000
        distances_3d_with_only_neighbours[i, j] = dist

geodesic_distances = shortest_path(distances_3d_with_only_neighbours, method='D')
g_dists = np.array(geodesic_distances)

gram_m = np.zeros((1000, 1000))
for i in range(len(gram_m)):
    for j in range(len(gram_m)):
        gram_m[i, j] = round((g_dists[0, j] ** 2 + g_dists[i, 0] ** 2 - g_dists[i, j] ** 2) / 2)

basis, _, _ = svd(gram_m)

X_2d = np.full((1000, 2), 0)

for i in range(len(X_2d)):
    X_2d[i, 0] = -np.dot(basis[:, 0], gram_m[:, i])
    X_2d[i, 1] = -np.dot(basis[:, 1], gram_m[:, i])

X_2d = (50 * X_2d) / X_2d.max()

fig = plt.figure(figsize=(16, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.rainbow)
ax.view_init(10, -80)
plt.title('Swiss Roll in 3D')
plt.savefig('SwissRoll.png')
plt.close()

iso = Isomap(n_components=2)

X_iso = iso.fit_transform(X)
plt.figure(figsize=(10, 6))
plt.scatter(X_iso[:, 0], X_iso[:, 1], c=color, cmap=plt.cm.rainbow)
plt.title('Standart Isomap')
plt.savefig('StandartIsoMap.png')
plt.close()

plt.figure(figsize=(10, 6))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=color, cmap=plt.cm.rainbow)
plt.title('My Isomap')
plt.savefig('MyIsoMap.png')
plt.show()
