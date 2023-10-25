import json
from sklearn.model_selection import GroupShuffleSplit
import numpy as np
import os
from constants import DIR_PATH


length = 1000
data_dir = os.path.join(DIR_PATH, f'updated_all_data_human_{length}/')
data_file = os.path.join(DIR_PATH, f"human_data_{length}_train.json")
chunk_size = 5000


def remove_outliers(data, percentage=0.02):
	print(np.median([sum(d['coding_seq']) / len(d['coding_seq']) for d in data]))
	return [d for d in data if 1-percentage >= (sum(d['coding_seq']) / len(d['coding_seq'])) >= percentage]


def split_data(data, chunk_size, prefix):
	for i in range(0, len(data), chunk_size):
		sd = {'data': data[i:i+chunk_size]}
		file_name = f'{data_dir}{prefix}_data_file_{int(i/chunk_size)}.json'
		with open(file_name, 'w') as fout:
			json.dump(sd, fout)


with open(data_file, 'r') as fin:
	all_data = json.load(fin)

print(len(all_data))
groups = [d['gene_id'] for d in all_data]
gss = GroupShuffleSplit(n_splits=10,  random_state=42)
for i, (train_index, test_index) in enumerate(gss.split(X=groups, y=groups, groups=groups)): # X, y are not important here
	train = [all_data[ind] for ind in train_index]
	val = [all_data[ind] for ind in test_index]
	break

split_data(train, chunk_size, 'train')
split_data(val, chunk_size, 'val')
