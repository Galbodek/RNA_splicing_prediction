from classification_model import MyHyenaDNA
from HyenaDNA import CharacterTokenizer
from create_dataset import get_train_val_test
import torch
import torch.nn as nn
import numpy as np
from comet_ml import Experiment
from utilities import clear_cache, get_loss
import json


pretrained_model_name = 'hyenadna-medium-450k-seqlen'
max_length = 450_000


def main(args):
    # set the device
    device = args.device
    use_cuda = (device != -1) and torch.cuda.is_available()
    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device is: {device}")

    # define experiment
    experiment = Experiment(api_key="OUSdYva9STAffLftJFYaSyah4", project_name="splices")
    experiment.add_tags(args.tags.split(','))
    parameters = {'batch_size': args.batch_size, 'learning_rate': args.lr}
    experiment.log_parameters(parameters, prefix='train')

    save_iter = args.save_interval  # next save

    # clear the cache
    clear_cache()

    model = MyHyenaDNA(pretrained_model_name, args.num_layers, args.h_dim, args.dropout).to(device)
    tokenizer = CharacterTokenizer(characters=['A', 'C', 'G', 'T', 'N'],  # add DNA characters
        model_max_length=max_length)
    train_dataset, test_dataset, val_dataset = get_train_val_test(args.data_file, tokenizer)
    train_generator = torch.utils.data.DataLoader(train_dataset, shuffle=True, num_workers=0, batch_size=args.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    curr_step = 0
    for epoch in range(1, args.epochs + 1):
        print(f'----- starting epoch = {epoch} -----')
        # Training
        model.train()
        for batch_num, (x, y) in enumerate(train_generator):
            loss, metrics = get_loss(model, x, y, device)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            loss_estimate = loss.item()

            # clip the gradients if needed
            if not np.isinf(args.clip):
                nn.utils.clip_grad_norm_(model.linear_relu_stack.parameters(), args.clip)

            # parameter update
            optimizer.step()

            experiment.log_confusion_matrix(matrix=metrics['confusion_matrix'])
            metrics['confusion_matrix'] = json.dumps(metrics['confusion_matrix'].tolist())
            experiment.log_metrics(metrics, step=batch_num+curr_step, epoch=epoch)
            experiment.log_metrics({"loss": loss_estimate}, step=batch_num+curr_step, epoch=epoch)

            # clear the cache
            clear_cache()

        model.eval()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Training a model for splicing classification')
    # training dataset
    parser.add_argument('--data_file', default='', help='path to the dataset train')
    # embedding model architecture
    parser.add_argument('--h_dim', type=int, default=128, help='dimension of hidden units of first linear layer')
    parser.add_argument('--num_layers', type=int, default=2, help='number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
    # training parameters
    parser.add_argument('-e', '--epochs', type=int, default=3, help='number of training epochs.')
    parser.add_argument('--save-interval', type=int, default=100000, help='number of step between data saving')
    parser.add_argument('--batch_size', type=int, default=64, help='minibatch size')
    parser.add_argument('--weight_decay', type=float, default=0, help='L2 regularization (default: 0)')
    parser.add_argument('--lr', type=float, default=0.00001, help='learning rate (default: 1e-4)')
    parser.add_argument('--clip', type=float, default=np.inf, help='gradient clipping max norm (default: inf)')
    parser.add_argument('--output', help='output file path')
    parser.add_argument('--save-prefix', help='path prefix for saving models')
    parser.add_argument('--tags', default='splicing', help='tags for comet-ml experiment')
    parser.add_argument('--device', type=int, default=-2, help='compute device to use')
    parser.add_argument('--debug', action='store_true')
    parser.set_defaults(debug=False)
    args = parser.parse_args()

    main(args)