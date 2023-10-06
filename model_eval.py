import argparse
import torch
import pickle
import os
import numpy as np
from classification_model import MyHyenaDNA
from HyenaDNA import CharacterTokenizer
from torch.utils.data import DataLoader
from train_model import pretrained_model_name, max_length
from create_dataset import get_test
from utilities import get_loss, collate_batch, clear_cache


def load_model(model_path, device):
    model = MyHyenaDNA.load_pretrained(model_path, pretrained_model_name, device)
    model = model.to(device)
    model.eval()
    return model


def main(args):
    # set the device
    print(torch.cuda.is_available())
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != -1 else "cpu")
    print(f"Device is: {device}")

    tokenizer = CharacterTokenizer(characters=['A', 'C', 'G', 'T', 'N'], model_max_length=max_length + 2, add_special_tokens=False)
    softmax = torch.nn.Softmax(dim=1)

    # clear the cache
    clear_cache()

    all_results = []
    for dir_path in args.data_dirs:
        dir_name = os.path.split(dir_path)[-1]
        species, length = dir_name.split("_")[1:3]
        model = load_model(f"{length}__{args.model_prefix}.sav", device)
        all_y = []
        all_y_probs = []
        test_dataset = get_test(dir_path, tokenizer, max_length)
        test_generator = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=lambda b: collate_batch(b, tokenizer))
        for batch_num, (x, y) in enumerate(test_generator):
            logits = model(x.to(device)).cpu()
            y_probs = softmax(logits, dim=2)
            all_y.append(y.numpy())
            all_y_probs.append(y_probs.numpy())

        labels = np.concatenate(all_y)
        probs = np.concatenate(all_y_probs)
        all_results.append({'species': species, 'length': length, 'labels': labels, 'probs': probs})

    with open(args.output_path, 'w') as fout:
        pickle.dump(all_results, fout)



if __name__ == "__main__":
    parser = argparse.ArgumentParser('Evaluating the model for splicing detection')
    # test datasets
    parser.add_argument("-d", "--data_dirs", nargs='*', type=str, help="The list of data directories to run on. should be '{prefix}_{species}_{length}")
    parser.add_argument("-o", "--output_path", type=str, help="path to ouput pickle file")

    # embedding model architecture
    parser.add_argument('-m', '--model_prefix', help='prefix to pretrained model, the path should look like "{prefix}__{length}.sav"')
    # training parameters - 1000000
    parser.add_argument('--batch-size', type=int, default=1, help='minibatch size')
    parser.add_argument('--tags', default='mge', help='tags for comet-ml experiment')
    parser.add_argument('--device', type=int, default=-1, help='compute device to use')
    args = parser.parse_args()

    main(args)