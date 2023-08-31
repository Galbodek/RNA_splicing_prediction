import json
from sklearn.model_selection import train_test_split


data_file = "/sternadi/nobackup/volume1/ellarannon/splicing/data_fil.json"
chunk_size = 1000

with open(data_file, 'r') as fin:
	all_data = json.load(fin)


def split_data(data, chunk_size, prefix):
	for i in range(0, len(data), chunk_size):
		sd = {'data': data[i:i+chunk_size]}
        file_name = f'./data/{prefix}_data_file_{int(i/chunk_size)}.json'
        with open(file_name, 'w') as fout:
        	json.dump(sd, fout)


train, test = train_test_split(d, test_size=0.2, random_state=42)
val, test = train_test_split(test, test_size=0.5, random_state=42)
split_data(train, chunk_size, 'train')
split_data(val, chunk_size, 'val')
split_data(test, chunk_size, 'test')