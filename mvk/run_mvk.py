from mvlearn.datasets import load_UCImultifeature
# from mvlearn.cluster import MultiviewKMeans
# from mv_kemans import MultiviewKMeans
from mv_kmeans_gpu import MultiviewKMeans
from sklearn.cluster import KMeans
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import normalized_mutual_info_score as nmi_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Load in UCI digits multiple feature data

RANDOM_SEED = 5

# Load dataset along with labels for digits 0 through 4
n_class = 10
Xs, labels = load_UCImultifeature(select_labeled=list(range(n_class)), views=[0, 1])
# Repeat data 10 times
Xs = [np.tile(X, (10, 1)) for X in Xs]
labels = np.tile(labels, 10)

print(Xs[0].shape)
print(labels.shape)

# Helper function to display data and the results of clustering
def display_plots(pre_title, data, labels):
    # plot the views
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    dot_size = 10
    ax[0].scatter(data[0][:, 0], data[0][:, 1], c=labels, s=dot_size)
    ax[0].set_title(pre_title + ' View 1')
    ax[0].axes.get_xaxis().set_visible(False)
    ax[0].axes.get_yaxis().set_visible(False)

    ax[1].scatter(data[1][:, 0], data[1][:, 1], c=labels, s=dot_size)
    ax[1].set_title(pre_title + ' View 2')
    ax[1].axes.get_xaxis().set_visible(False)
    ax[1].axes.get_yaxis().set_visible(False)

    plt.show()

m_kmeans = MultiviewKMeans(n_clusters=n_class, random_state=RANDOM_SEED)
m_clusters = m_kmeans.fit_predict(Xs)

# Compute nmi between true class labels and multiview cluster labels
m_nmi = nmi_score(labels, m_clusters)
print('Multiview NMI Score: {0:.3f}\n'.format(m_nmi))