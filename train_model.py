from comet_ml import Experiment
from classification_model import MyHyenaDNA
from HyenaDNA import CharacterTokenizer
from create_dataset import get_train_val_test
import torch
import torch.nn as nn
import numpy as np
from utilities import clear_cache, get_loss, collate_batch
import json


pretrained_model_name = 'hyenadna-medium-160k-seqlen'
max_length = 160000


def define_experiment(args):
    experiment = Experiment(api_key="OUSdYva9STAffLftJFYaSyah4", project_name="splicing")
    experiment.add_tags(args.tags.split(','))
    parameters = {'batch_size': args.batch_size, 'learning_rate': args.lr}
    experiment.log_parameters(parameters, prefix='train')
    return experiment


def log_experiment(experiment, metrics, loss, step, epoch):
    experiment.log_confusion_matrix(matrix=metrics.pop('confusion_matrix'), step=step, epoch=epoch)
    # metrics['confusion_matrix'] = json.dumps(metrics['confusion_matrix'].tolist())
    experiment.log_metrics(metrics, step=step, epoch=epoch)
    experiment.log_metrics({"loss": loss}, step=step, epoch=epoch)


def save_model(model, save_prefix, batch_num, epoch, device):
    if save_prefix is not None:
        model.eval()
        save_path = f"{save_prefix}__epoch__{epoch}__iter__{batch_num}.sav"
        model.cpu()
        torch.save(model.state_dict(), save_path)
        model.to(device)

        # flip back to train mode
        model.train()


def main(args):
    # set the device
    print(torch.cuda.is_available())
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != -1 else "cpu")
    print(f"Device is: {device}")
    save_prefix = f'{args.save_prefix}__{args.h_dim}__{args.num_layers}__{args.dropout}'


    # define experiment
    experiment = define_experiment(args)

    # clear the cache
    clear_cache()

    # initialize the model
    if args.model is not None:
        # load pretrained model
        model = torch.load(args.model)
    else:
        model = MyHyenaDNA(pretrained_model_name, device, args.num_layers, args.h_dim, args.dropout).to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"Total number of parameters in model: {sum(p.numel() for p in params)}")
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    tokenizer = CharacterTokenizer(characters=['A', 'C', 'G', 'T', 'N'], model_max_length=max_length + 2,  # to account for special tokens, like EOS
                                   add_special_tokens=False)  # we handle special tokens elsewhere

    train_dataset, test_dataset, val_dataset = get_train_val_test(args.data_file, tokenizer, max_length)
    train_generator = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=lambda b: collate_batch(b, tokenizer))
    curr_step = 0
    save_iter = args.save_interval  # next save
    for epoch in range(1, args.epochs + 1):
        print(f'----- starting epoch = {epoch} -----')
        # Training
        model.train()
        # train_dataset.set_epoch(epoch) # Shuffling data for each epoch
        for batch_num, (x, y) in enumerate(train_generator):
            # clear the cache
            clear_cache()
            loss, metrics = get_loss(model, x, y, device)
            loss = loss / args.accum_iter # normalize loss to account for batch accumulation


            # Backpropagation
            loss.backward()
            loss_estimate = loss.item()


             # weights update
            if ((batch_num + 1) % args.accum_iter == 0) or (batch_num + 1 == len(train_generator)):
                optimizer.step()
                optimizer.zero_grad()

            # clip the gradients if needed
            if not np.isinf(args.clip):
                nn.utils.clip_grad_norm_(model.classification_head.parameters(), args.clip)

            log_experiment(experiment, metrics, loss_estimate*args.accum_iter, batch_num + curr_step, epoch)

            # clear the cache
            clear_cache()

            if batch_num == save_iter:
                save_iter = save_iter + args.save_interval  # next save
                save_model(model, save_prefix, batch_num, epoch, device)

        curr_step += batch_num

    save_model(model, save_prefix, batch_num, epoch, device)
    experiment.end()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Training a model for splicing classification')
    # training dataset
    parser.add_argument('--data_file', default='', help='path to the dataset train')
    # embedding model architecture
    parser.add_argument('model', nargs='?', help='pretrained model (optional)')
    parser.add_argument('--h_dim', type=int, default=128, help='dimension of hidden units of first linear layer')
    parser.add_argument('--num_layers', type=int, default=2, help='number of linear layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
    # training parameters
    parser.add_argument('-e', '--epochs', type=int, default=3, help='number of training epochs.')
    parser.add_argument('--save-interval', type=int, default=1000, help='number of step between data saving')
    parser.add_argument('--batch_size', type=int, default=64, help='minibatch size')
    parser.add_argument('-ac', '--accum_iter', type=int, default=16, help='number of batches we calculate before updating gradients.')
    parser.add_argument('--weight_decay', type=float, default=0, help='L2 regularization (default: 0)')
    parser.add_argument('--lr', type=float, default=0.00001, help='learning rate (default: 1e-4)')
    parser.add_argument('--clip', type=float, default=np.inf, help='gradient clipping max norm (default: inf)')
    parser.add_argument('--output', help='output file path')
    parser.add_argument('--save-prefix', help='path prefix for saving models')
    parser.add_argument('--tags', default='splicing', help='tags for comet-ml experiment')
    parser.add_argument('--device', type=int, default=-1, help='compute device to use')
    parser.add_argument('--debug', action='store_true')
    parser.set_defaults(debug=False)
    args = parser.parse_args()

    main(args)