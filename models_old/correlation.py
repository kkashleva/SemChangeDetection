import argparse
import numpy as np
import pickle
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

n_clusters = 8


def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--embeddings_path', type=str, help="Path to a file with embeddings")
    arg_parser.add_argument('--dataset_path', type=str, help="Path to a dataset for correlation calculation")
    args = arg_parser.parse_args()
    if not args.embeddings_path:
        print('Please specify path to embeddings (--embeddings_path)')
    elif not args.dataset_path:
        print('Please specify path to a dataset ; (--dataset_path)')
    else:
        return args


def cluster_word_embeddings_k_means(word_embeddings):
    if len(word_embeddings) > n_clusters:
        clustering = KMeans(random_state=0, n_clusters=n_clusters).fit(word_embeddings)
        centroids = clustering.cluster_centers_
    else:
        centroids = word_embeddings
    return centroids


def normalize_means(data):
    normalized = [(x - min(data)) / (max(data) - min(data)) for x in data]
    return normalized


arguments = parse_args()
if not arguments:
    exit(1)
with open(arguments.embeddings_path, 'rb') as f:
    embeddings = pickle.load(f)
with open(arguments.dataset_path, encoding='utf-8-sig') as f:
    dataset = [i.split() for i in f.readlines()]
graded = {i[0]: i[1] for i in dataset}
time_epochs = list(embeddings.keys())
X = []
y = []
words = []
for i, word in enumerate(embeddings[time_epochs[0]]):
    print(f'Processing word {word} ({i + 1} out of {len(embeddings[time_epochs[0]])})')
    if word not in embeddings[time_epochs[1]]:
        continue
    centroids_first = cluster_word_embeddings_k_means(embeddings[time_epochs[0]][word])
    centroids_second = cluster_word_embeddings_k_means(embeddings[time_epochs[1]][word])
    try:
        cosine_change = cdist(centroids_first, centroids_second)
        mean = np.mean(cosine_change)
    except ValueError:
        continue
    X.append(mean)
    y.append(float(graded[word]))
    words.append(word)
    print(f'{mean} - {float(graded[word])}')
normalized_means = normalize_means(X)
print('After normalization:')
for normalized_mean, gold_standard, word in zip(normalized_means, y, words):
    print(f'{word}: {normalized_mean} - {gold_standard}')
print(stats.spearmanr(a=normalized_means, b=y))
