from datasets import load_dataset, DatasetDict
import torch.nn as nn
import numpy as np
import torch
import glob
import os


PROJECT_DIR = '/davidb/ellarannon/splicing/' #'/sternadi/nobackup/volume1/ellarannon/splicing/'


class ClassificationDataset(torch.utils.data.Dataset): # IterableDataset
    def __init__(self, dataset):
        self.dataset = dataset

    def __process_example(self, example):
        X, y = torch.Tensor(example['input_ids']).squeeze(0).to(dtype=torch.int32), example['label']#torch.LongTensor(example['labels']).squeeze(0)
        return X, y

    def __getitem__(self, idx):
        return self.__process_example(self.dataset[idx])

    def __iter__(self):
        for example in self.dataset:
            yield self.__process_example(example)

    def __len__(self):
        return len(self.dataset)

    # def set_epoch(self, epoch):
    #     self.dataset.set_epoch(epoch)


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
          # padding='max_length', # Padding in collate batch
          truncation=True,
          max_length=max_length,
          # return_attention_mask=True,
          # return_token_type_ids=True,
          return_tensors='pt')

    # # adding padding (to max len) to the left like the tokenizer
    # output = nn.ConstantPad1d((max_length - len(data['coding_seq']), 0), 0)(torch.LongTensor(data['coding_seq']))
    # return {'input_ids': input, 'labels': output}
    # return {'input_ids': input, 'labels': torch.LongTensor(data['coding_seq'][:max_length]), 'weight': get_weight(data['coding_seq'][:max_length])}
    return {'input_ids': input, 'label': data['class'], 'weight': 1} # data['coding_seq'] # TODO see if we need to change weight


def get_train_val_test(data_dir, tokenizer, max_length):
    file_mapping = {'train': glob.glob(f"{data_dir}/train_data_file_*.json"), 'test': glob.glob(f"{data_dir}/test_data_file_*.json"), "valid":glob.glob(f"{data_dir}/val_data_file_*.json")}
    dataset = load_dataset('json', data_files=file_mapping, cache_dir=os.path.join(PROJECT_DIR, "cache"), field='data') # , streaming=True
    # dataset = split_dataset(dataset)
    train = dataset['train'].map(lambda x: encode(x, tokenizer, max_length), remove_columns=['unspliced_transcript', 'class'])
    train_weights = torch.Tensor([s['weight'] for s in train])
    # test set will be loaded individualy in other func
    # test = dataset['test'].map(lambda x: encode(x, tokenizer, max_length), remove_columns=['unspliced_transcript', 'coding_seq'])
    val = dataset['valid'].map(lambda x: encode(x, tokenizer, max_length), remove_columns=['unspliced_transcript', 'class'])
    train_dataset, val_dataset = ClassificationDataset(train), ClassificationDataset(val)
    return train_dataset, val_dataset, train_weights


def get_test(data_dir, tokenizer, max_length):
    file_mapping = {'test': glob.glob(f"{data_dir}/test_data_file_*.json")}
    dataset = load_dataset('json', data_files=file_mapping, cache_dir=os.path.join(PROJECT_DIR, "cache"), field='data') # , streaming=True
    test = dataset['test'].map(lambda x: encode(x, tokenizer, max_length), remove_columns=['unspliced_transcript', 'class'])
    test_dataset = ClassificationDataset(test)
    return test_dataset
