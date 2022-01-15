import argparse
import pickle
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AffinityPropagation


def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--embeddings_path', type=str, help="Path to a file with embeddings")
    arg_parser.add_argument('--dataset_path', type=str, help="Path to a dataset for correlation calculation")
    args = arg_parser.parse_args()
    if not args.embeddings_path:
        print('Please specify path to embeddings (--embeddings_path)')
    elif not args.dataset_path:
        print('Please specify path to a dataset ; (--dataset_path)')
    return args


def cluster_word_embeddings_aff_prop(word_embeddings):
    clustering = AffinityPropagation().fit(word_embeddings)
    centroids = clustering.cluster_centers_
    return centroids


arguments = parse_args()
if not arguments:
    exit(1)
with open(arguments.embeddings_path, 'rb') as f:
    embeddings = pickle.load(f)
with open(arguments.dataset_path) as f:
    dataset = [i.split() for i in f.readlines()]
graded = {i[0][:-3]: i[1] for i in dataset}
time_epochs = list(embeddings.keys())
X = []
y = []
for i, word in enumerate(embeddings[time_epochs[0]]):
    print(f'Processing word {word} ({i + 1} out of {len(embeddings[time_epochs[0]])})')
    if word not in embeddings[time_epochs[1]]:
        continue
    centroids_first = cluster_word_embeddings_aff_prop(embeddings[time_epochs[0]][word])
    centroids_second = cluster_word_embeddings_aff_prop(embeddings[time_epochs[1]][word])
    average_centroid_first = centroids_first.mean(axis=0)
    average_centroid_second = centroids_second.mean(axis=0)
    try:
        cosine_change = (1 - cosine_similarity([average_centroid_first], [average_centroid_second]))[0][0]
    except ValueError:
        continue
    X.append(cosine_change)
    y.append(float(graded[word]))
    print(f'{cosine_change} - {float(graded[word])}')
print(stats.spearmanr(a=X, b=y))
