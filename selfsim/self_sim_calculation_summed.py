import argparse
import pickle
import csv
import numpy as np
import random
from sklearn.metrics.pairwise import pairwise_distances

reps = [] # a list to store representations
with (open("bert-base-uncased_12layers.pickle", "rb")) as openfile:
    while True:
        try:
            reps = pickle.load(openfile)
        except EOFError:
            break

reps = reps['Processed/semeval2020_ulscd_eng/processed_ccoha2.txt']

rows = []
for word in reps.keys():
    row = []
    row.append(word)
    row.append(len(reps[word]))
    pairwise = pairwise_distances(reps[word], # calculate cosine distances and take upper triangular
        metric='cosine')[np.triu_indices(reps[word].shape[0], k=1)]
    # turn cosine distances into similarities
    cosine_list = [1 - c for c in pairwise]
    average_cosine = np.average(cosine_list)
    row.append(average_cosine)
    rows.append(row)
    print(f'self-sim was calculated for {word}')

head = ['word','instances','self-sim']

with open('self_sim_summed_ccoha2.csv', 'w', newline='') as csvfile:
     writer = csv.writer(csvfile, delimiter=',')
     writer.writerow(head)
     for row in rows:
        writer.writerow(row)