from datasets import load_dataset, DatasetDict
import torch.nn as nn
import torch


class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        return torch.Tensor(self.dataset[idx]['input_ids']).squeeze(0).to(dtype=torch.int32), torch.Tensor(self.dataset[idx]['labels']).squeeze(0).to(dtype=torch.int32)

    def __len__(self):
        return len(self.dataset)


def split_dataset(dataset):
    train_test_valid = dataset.train_test_split(test_size=0.4)
    test_valid = train_test_valid['test'].train_test_split(test_size=0.5)

    dataset = DatasetDict({
        'train': train_test_valid['train'],
        'test': test_valid['test'],
        'valid': test_valid['train']})
    return dataset


def encode(data, tokenizer):
  max_length = 45
  input = tokenizer.encode(data['unspliced_transcript'],
          add_special_tokens=True,
          padding='max_length',
          truncation=True,
          max_length=max_length,
          return_attention_mask=True,
          return_token_type_ids=True,
          return_tensors='pt')

  # adding padding (to max len) to the left like the tokenizer
  output = nn.ConstantPad1d((max_length - len(data['coding_seq']), 0), 0)(torch.LongTensor(data['coding_seq']))
  return {'input_ids': input, 'labels': output}


def get_train_val_test(data_file, tokenizer):
    dataset = load_dataset('json', data_files=data_file)
    dataset = split_dataset(dataset)
    train = dataset['train'].map(lambda x: encode(x, tokenizer), remove_columns=['unspliced_transcript', 'coding_seq'])
    test = dataset['test'].map(lambda x: encode(x, tokenizer), remove_columns=['unspliced_transcript', 'coding_seq'])
    val = dataset['valid'].map(lambda x: encode(x, tokenizer), remove_columns=['unspliced_transcript', 'coding_seq'])
    train_dataset, test_dataset, val_dataset = ClassificationDataset(train), ClassificationDataset(test), ClassificationDataset(val)
    return train_dataset, test_dataset, val_dataset
