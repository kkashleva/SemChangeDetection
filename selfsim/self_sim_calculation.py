import argparse
import pickle
import csv
import numpy as np
import random
from sklearn.metrics.pairwise import pairwise_distances

# код пока не закодументирован
# пока просто как скрипт чтобы быстро получить результат

def parse_args():
	# позже добавлю аргументы, чтобы можно было считать для любого корпуса
	pass 

reps = [] # a list to store representations
with (open("bert-base-uncased_12layers_concatenated_ccoha2.pickle", "rb")) as openfile:
    while True:
        try:
            reps = pickle.load(openfile)
        except EOFError:
            break

reps = reps['processed_ccoha1.txt']

def get_reps_by_layer(word_reps):
	"""
	Получаем эмбеддинги для всех словоупотреблений word_reps
	по слоям.
	word_reps -> список конкатенированных эмбеддигнов длиной n,
	где n - число словоупотреблений слова w в корпусе
	"""

	result = [[], [], [], [], [], [], [], [], [], [], [], []]
	# проходимся по конкатенированному эмбеддингу для каждого токена (9216 значений)
	for token_reps in word_reps:
		# делим на 12 списков из 768 значений
		token_reps = np.split(token_reps, 12)
		for i in range(0, 12):
			# добавляем эмбеддинг 
			result[i].append(token_reps[i])

	return(np.array(result))

def calculate_self_sim(word_reps_by_layer, layer):
	"""  
	Расчет метрики Self-similarity для слова с 
	заданного слоя layer
	"""
	
	word_reps_by_layer = word_reps_by_layer[layer-1]
	# код ниже взят из проекта monopoly
	my_pairwise = pairwise_distances(word_reps_by_layer, # calculate cosine distances and take upper triangular
		metric='cosine')[np.triu_indices(word_reps_by_layer.shape[0], k=1)]
	# turn cosine distances into similarities
	cosine_list = [1 - c for c in my_pairwise]
	average_cosine = np.average(cosine_list)

	return average_cosine

rows = []

for word in reps.keys():
	row = []
	row.append(word)
	row.append(len(reps[word]))
	reps_by_layer = get_reps_by_layer(reps[word])
	for i in range(1,13):
		row.append(calculate_self_sim(reps_by_layer, i))
	rows.append(row)
	print(f'self-sim was calculated for {word}')

head = ['word','instances',1,2,3,4,5,6,7,8,9,10,11,12]

with open('self_sim_by_layer_ccoha1.csv', 'w', newline='') as csvfile: # output path
	 writer = csv.writer(csvfile, delimiter=',')
	 writer.writerow(head)
	 for row in rows:
	 	writer.writerow(row)



