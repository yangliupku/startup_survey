from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from sklearn.utils import resample
import numpy as np
import seaborn as sns
import itertools
import matplotlib.pyplot as plt


def print_top_words(model, feature_names, n_top_words=20):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()

from nltk.stem import WordNetLemmatizer
import re
token_pattern = re.compile(r"(?u)\b\w\w+\b")


class LemmaTokenizer(object):

    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in token_pattern.findall(doc)]


def plot_top_words(model, feature_names, topic_idx, n_top_words=10):
    #     for topic_idx, topic in enumerate(model.components_):
    topic = normalize(model.components_[topic_idx].reshape(1, -1))[0]
    sorted_idx = topic.argsort()[:-n_top_words - 1:-1]
    print("Topic #%d:" % topic_idx)
    features = [feature_names[i] for i in sorted_idx]
    print(" ".join(features))
    ax = sns.barplot(x=list(range(10)), y=topic[
                     sorted_idx], palette=sns.color_palette('muted'))
    feature_id = 0
    height = max(topic[sorted_idx]) * 0.6

    for p in ax.patches:
        ax.text(p.get_x() + 0.15, height - 0.1,
                '{}'.format(features[feature_id]), fontsize=10)
        feature_id += 1


def plot_cluster(model, features, tsne_features, feature_names, nmf_model):
    """
    model: the clustering model. model.labels_ should contain the clusting label
    features: feature vectors used to compute the clustering centroid
    tsne_features: feature vectors for plot with tsne
    """
    cluster_labels = model.labels_
    n_top_words = 10
    num_clusters = max(cluster_labels) + 1
    # plotting
    # sample 2000 data points for scattering plot
    palette = itertools.cycle(sns.color_palette("muted"))
    markers = itertools.cycle(['x', 'o', 'v', '^', '<', 's'])

    # sample_for_plt, label_for_plt = resample(
    #     tsne_features, cluster_labels, n_samples=4000, random_state=0)
    mapping = TSNE(n_components=2, init='pca', random_state=0,
                   n_iter=10000, verbose=0, learning_rate=20, perplexity=40)
    embed = mapping.fit_transform(tsne_features)
    fig = plt.figure(figsize=(12, 5))
    # plot the clusters
    plt.subplot(1, 2, 1)
    for i in range(num_clusters):
        subgroup = embed[cluster_labels == i, :]
        plt.scatter(subgroup[:, 0], subgroup[:, 1], s=15, color=next(
            palette), marker=next(markers), label='{}'.format(i))
    plt.legend()
    # plot the histgram of the clusters
    plt.subplot(1, 2, 2)
    plt.hist(cluster_labels, bins=num_clusters)
    # print out the cluster centers
    fig = plt.figure(figsize=(12, 8))
    for i in range(max(cluster_labels) + 1):
        cluster = features[cluster_labels == i, :]
        centroid = np.mean(cluster, axis=0)
        centroid_feature = centroid.dot(nmf_model.components_)
        sorted_idx = centroid_feature.argsort()[:-n_top_words - 1:-1]
        plt.subplot(num_clusters, 1, i + 1)
        ax = sns.barplot(x=list(range(10)), y=centroid_feature[
                         sorted_idx], palette=sns.color_palette('muted'))
        top_words = [feature_names[j] for j in sorted_idx]
        word_id = 0
        height = max(centroid_feature[sorted_idx]) * 0.6
        for p in ax.patches:
            ax.text(p.get_x() + 0.15, height,
                    '{}'.format(top_words[word_id]), fontsize=10)
            word_id += 1


def print_cluster_member(data, cluster_id, num_samples=10):
    cluster_member = data[data['label'] == cluster_id]
    print('Class:', cluster_member.iloc[0]['label_name'])
    print('number of companies in this class:', len(cluster_member))
    print('-' * 10, 'Sample Startups', '-' * 10)
    for m in cluster_member.head(num_samples).iterrows():
        print('company ID:', m[0])
        print(m[1]['google_description'])
