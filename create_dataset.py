from datasets import load_dataset, DatasetDict
from constants import DIR_PATH
import torch.nn as nn
import numpy as np
import torch
import glob
import os


class ClassificationDataset(torch.utils.data.Dataset): # IterableDataset
    def __init__(self, dataset):
        self.dataset = dataset

    def __process_example(self, example):
        X, y = torch.Tensor(example['input_ids']).squeeze(0).to(dtype=torch.int32), example['label']
        return X, y

    def __getitem__(self, idx):
        return self.__process_example(self.dataset[idx])

    def __iter__(self):
        for example in self.dataset:
            yield self.__process_example(example)

    def __len__(self):
        return len(self.dataset)


def get_weight(labels, epsilon=0.1, max_weight=0.1):
    percentage = np.mean(labels)
    return min(percentage + epsilon, max_weight)


def split_dataset(dataset):
    train_test_valid = dataset.train_test_split(test_size=0.4)
    test_valid = train_test_valid['test'].train_test_split(test_size=0.5)

    dataset = DatasetDict({
        'train': train_test_valid['train'],
        'test': test_valid['test'],
        'valid': test_valid['train']})
    return dataset


def encode(data, tokenizer, max_length):
    input = tokenizer.encode(data['unspliced_transcript'],
          add_special_tokens=False,
          truncation=True,
          max_length=max_length,
          return_tensors='pt')

    return {'input_ids': input, 'label': data['class'], 'weight': 1} # data['coding_seq'] # TODO see if we need to change weight


def get_train_val_test(data_dir, tokenizer, max_length, train_file_num=0):
    file_mapping = {'train': glob.glob(f"{data_dir}/train_data_file_*.json"), "valid":glob.glob(f"{data_dir}/val_data_file_*.json")}
    if train_file_num > 0:
        file_mapping['train'] = [file_path for file_path in file_mapping['train'] if int(file_path.split('train_data_file_')[1].replace(".json", "")) < train_file_num]
    dataset = load_dataset('json', data_files=file_mapping, cache_dir=os.path.join(DIR_PATH, "cache"), field='data') # , streaming=True
    train = dataset['train'].map(lambda x: encode(x, tokenizer, max_length), remove_columns=['unspliced_transcript', 'class'])
    train_weights = torch.Tensor([s['weight'] for s in train])
    val = dataset['valid'].map(lambda x: encode(x, tokenizer, max_length), remove_columns=['unspliced_transcript', 'class'])
    train_dataset, val_dataset = ClassificationDataset(train), ClassificationDataset(val)
    return train_dataset, val_dataset, train_weights


def get_validation(data_dir, tokenizer, max_length, val_file_num=0):
    file_mapping = {'valid': glob.glob(f"{data_dir}/val_data_file_*.json")}
    if val_file_num > 0:
        file_mapping['valid'] = [file_path for file_path in file_mapping['valid'] if int(file_path.split('val_data_file_')[1].replace(".json", "")) < val_file_num]

    dataset = load_dataset('json', data_files=file_mapping, cache_dir=os.path.join(DIR_PATH, "cache"), field='data') # , streaming=True
    val = dataset['valid'].map(lambda x: encode(x, tokenizer, max_length), remove_columns=['unspliced_transcript', 'class'])
    val_dataset = ClassificationDataset(val)
    return val_dataset


def get_test(data_dir, tokenizer, max_length):
    file_mapping = {'test': glob.glob(f"{data_dir}/test_data_file_*.json")}
    dataset = load_dataset('json', data_files=file_mapping, cache_dir=os.path.join(DIR_PATH, "cache"), field='data') # , streaming=True
    test = dataset['test'].map(lambda x: encode(x, tokenizer, max_length), remove_columns=['unspliced_transcript', 'class'])
    test_dataset = ClassificationDataset(test)
    return test_dataset
