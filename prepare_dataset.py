import json
from sklearn.model_selection import train_test_split
import numpy as np


def remove_outliers(data, percentage=0.02):
	print(np.median([sum(d['coding_seq']) / len(d['coding_seq']) for d in data]))
	return [d for d in data if 1-percentage >= (sum(d['coding_seq']) / len(d['coding_seq'])) >= percentage]


def split_data(data, chunk_size, prefix):
	for i in range(0, len(data), chunk_size):
		sd = {'data': data[i:i+chunk_size]}
		file_name = f'/sternadi/nobackup/volume1/ellarannon/splicing/data/{prefix}_data_file_{int(i/chunk_size)}.json'
		with open(file_name, 'w') as fout:
			json.dump(sd, fout)



data_file = "/sternadi/nobackup/volume1/ellarannon/splicing/encoded_transcripts.json"
chunk_size = 500

with open(data_file, 'r') as fin:
	all_data = json.load(fin)

print(len(all_data))
all_data = remove_outliers(all_data)
print(len(all_data))
train, test = train_test_split(all_data, test_size=0.2, random_state=42)
val, test = train_test_split(test, test_size=0.5, random_state=42)
split_data(train, chunk_size, 'train')
split_data(val, chunk_size, 'val')
split_data(test, chunk_size, 'test')