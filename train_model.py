from comet_ml import Experiment
from classification_model import MyHyenaDNA
from HyenaDNA import CharacterTokenizer
from create_dataset import get_train_val_test
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn as nn
import numpy as np
from utilities import clear_cache, get_loss, collate_batch, compute_metrics


pretrained_model_name = 'hyenadna-medium-160k-seqlen'
n_files = 191  # number of file in the 10K case
max_length = 160000


def define_experiment(args):
    experiment = Experiment(api_key="OUSdYva9STAffLftJFYaSyah4", project_name="splicing")
    experiment.add_tags(args.tags.split(','))
    parameters = {'batch_size': args.batch_size, 'learning_rate': args.lr}
    experiment.log_parameters(parameters, prefix='train')
    return experiment


def log_experiment(experiment, metrics, step, epoch):
    experiment.log_confusion_matrix(matrix=metrics.pop('confusion_matrix'), step=step, epoch=epoch)
    experiment.log_metrics(metrics, step=step, epoch=epoch)


def save_model(model, save_prefix, batch_num, epoch, device):
    if save_prefix is not None:
        model.eval()
        save_path = f"{save_prefix}__epoch__{epoch}__iter__{batch_num}.sav"
        model.cpu()
        torch.save(model.state_dict(), save_path)
        model.to(device)

        # flip back to train mode
        model.train()


def run_model_on_val(model, dataset, device, batch_size, tokenizer):
    model.eval()
    all_y, all_logits = [] , []
    data_generator = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda b: collate_batch(b, tokenizer))
    for batch_num, (x, y) in enumerate(data_generator):
        logits = model(x.to(device))
        all_y.append(y.cpu().detach())
        all_logits.append(logits.cpu().detach())

    labels = torch.concatenate(all_y, dim=0)
    logits = torch.concatenate(all_logits, dim=0)
    print(compute_metrics(labels, logits))
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
        model = MyHyenaDNA.load_pretrained(args.model, pretrained_model_name, device)
        model = model.to(device)
    else:
        model = MyHyenaDNA(pretrained_model_name, device, args.num_layers, args.h_dim, args.dropout).to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"Total number of parameters in model: {sum(p.numel() for p in params)}")
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    tokenizer = CharacterTokenizer(characters=['A', 'C', 'G', 'T', 'N'], model_max_length=max_length + 2,  # to account for special tokens, like EOS
                                   add_special_tokens=False)  # we handle special tokens elsewhere

    train_dataset, val_dataset, train_weights = get_train_val_test(args.data_file, tokenizer, max_length, train_file_num=n_files)
    print(f"mean_percentage: {torch.mean(train_weights)}")
    mean_percentage = 0.5 # min(torch.mean(train_weights) * 8, 0.5)  # reducing the effect on loss function
    class_weights = [1/(1-mean_percentage), 1/mean_percentage] #[1.136, 8.34] # The median of exon proportion is ~12%, therefore weights is 1/0.88 and 1/0.12
    class_weights = torch.tensor(class_weights).to(device)

    train_generator = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=lambda b: collate_batch(b, tokenizer))
    curr_step = 0
    for epoch in range(1, args.epochs + 1):
        print(f'----- starting epoch = {epoch} -----')
        save_iter = args.save_interval  # next save
        # Training
        model.train()
        # train_dataset.set_epoch(epoch) # Shuffling data for each epoch
        for batch_num, (x, y) in enumerate(train_generator):
            # clear the cache
            clear_cache()
            loss, metrics = get_loss(model, x, y, class_weights, device)
            loss = loss / args.accum_iter # normalize loss to account for batch accumulation

            # Backpropagation
            loss.backward()
            loss_estimate = loss.item()

             # weights update
            if ((batch_num + 1) % args.accum_iter == 0) or (batch_num + 1 == len(train_generator)):
                optimizer.step()
                optimizer.zero_grad()
                experiment.log_metrics({"loss": loss_estimate*args.accum_iter}, step=round((batch_num + curr_step) / args.accum_iter), epoch=epoch)

            # clip the gradients if needed
            if not np.isinf(args.clip):
                nn.utils.clip_grad_norm_(model.classification_head.parameters(), args.clip)

            log_experiment(experiment, metrics, batch_num + curr_step, epoch)

            # clear the cache
            clear_cache()

            if batch_num == save_iter:
                save_iter = save_iter + args.save_interval  # next save
                save_model(model, save_prefix, batch_num, epoch, device)

        curr_step += batch_num
        save_model(model, save_prefix, batch_num, epoch, device)

    save_model(model, save_prefix, batch_num, epoch, device)
    print('finished training')
    # Running Model On Validation Set
    run_model_on_val(model, val_dataset, device, args.batch_size, tokenizer)
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
    parser.add_argument('--n_iter', type=int, default=0, help='number of batches to run per epoch. deafult: 0 (size of dataset)')
    parser.add_argument('--batch_size', type=int, default=64, help='minibatch size')
    parser.add_argument('-ac', '--accum_iter', type=int, default=16, help='number of batches we calculate before updating gradients.')
    parser.add_argument('--weight_decay', type=float, default=0, help='L2 regularization (default: 0)')
    parser.add_argument('--lr', type=float, default=0.00001, help='learning rate (default: 1e-4)')
    parser.add_argument('--clip', type=float, default=np.inf, help='gradient clipping max norm (default: inf)')
    parser.add_argument('--output', help='output file path')
    parser.add_argument('--save-prefix', help='path prefix for saving models')
    parser.add_argument('--tags', default='splicing', help='tags for comet-ml experiment')
    parser.add_argument('--device', type=int, default=-1, help='compute device to use')
    parser.set_defaults(debug=False)
    args = parser.parse_args()

    main(args)