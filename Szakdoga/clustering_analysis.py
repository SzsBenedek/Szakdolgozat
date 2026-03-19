import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from gui import load_data, FEATURE_FUNCS


def run_clustering(feature_name):
    # Adatok betöltése
    X, y = load_data(FEATURE_FUNCS[feature_name])

    # Normalizálás (nagyon fontos!)
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    # PCA 2D-re
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    # KMeans klaszterezés (2 klaszter)
    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(X)

    # Sziluett mérőszám (mennyire jó a szétválás)
    sil_score = silhouette_score(X, clusters)
    print(f"Silhouette score: {sil_score:.3f}")

    # Vizualizáció
    plt.figure(figsize=(8, 6))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=clusters)
    plt.title("EEG adatok klaszterezése (PCA + KMeans)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    run_clustering("Simple features")
