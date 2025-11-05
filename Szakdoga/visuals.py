import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from Main import load_data

X, y = load_data('data')
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)
plt.figure(figsize=(8,6))
plt.scatter(X_2d[y==0,0], X_2d[y==0,1], c='green', label='Kontroll', alpha=0.6)
plt.scatter(X_2d[y==1,0], X_2d[y==1,1], c='red', label='Beteg', alpha=0.6)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('EEG feature tér – PCA redukció')
plt.legend()
plt.grid(True)
plt.show()