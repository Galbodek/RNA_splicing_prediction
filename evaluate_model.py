from comet_ml import Experiment
from classification_model import MyHyenaDNA
from HyenaDNA import CharacterTokenizer
from create_dataset import get_test, get_validation
import torch
from torch.utils.data import DataLoader
from utilities import clear_cache, get_loss, collate_batch, compute_metrics
from train_model import define_experiment, log_experiment


pretrained_model_name = 'hyenadna-medium-160k-seqlen'
max_length = 160000


def main(args):
    # set the device
    print(torch.cuda.is_available())
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != -1 else "cpu")
    print(f"Device is: {device}")

    # define experiment
    experiment = define_experiment(args)

    # clear the cache
    clear_cache()

    # initialize the model
    model = MyHyenaDNA.load_pretrained(args.model, pretrained_model_name, device)
    model = model.to(device)
    model.eval()

    tokenizer = CharacterTokenizer(characters=['A', 'C', 'G', 'T', 'N'], model_max_length=max_length + 2,  # to account for special tokens, like EOS
                                   add_special_tokens=False)  # we handle special tokens elsewhere

    dataset = get_test(args.data_file, tokenizer, max_length, args.n_files) if args.test else get_validation(args.data_file, tokenizer, max_length, args.n_files)
    mean_percentage = 0.5 # min(torch.mean(train_weights) * 8, 0.5)  # reducing the effect on loss function
    class_weights = [1/(1-mean_percentage), 1/mean_percentage] #[1.136, 8.34] # The median of exon proportion is ~12%, therefore weights is 1/0.88 and 1/0.12
    class_weights = torch.tensor(class_weights).to(device)

    all_y, all_logits = [], []
    data_generator = DataLoader(dataset, batch_size=args.batch_size, collate_fn=lambda b: collate_batch(b, tokenizer))
    for batch_num, (x, y) in enumerate(data_generator):
        # clear the cache
        clear_cache()
        logits = model(x.to(device))
        all_y.append(y.cpu().detach())
        all_logits.append(logits.cpu().detach())

        loss, metrics = get_loss(model, x, y, class_weights, device)
        loss_estimate = loss.item()
        experiment.log_metrics({"loss": loss_estimate}, step=batch_num, epoch=0)
        log_experiment(experiment, metrics, batch_num, 0)

        # clear the cache
        clear_cache()

    labels = torch.concatenate(all_y, dim=0)
    logits = torch.concatenate(all_logits, dim=0)
    print(compute_metrics(labels, logits))
    experiment.end()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Evaluating the model for splicing classification')
    parser.add_argument('--data_file', default='', help='path to the dataset train')
    parser.add_argument('--model', help='path to pretrained model')
    parser.add_argument('-nf', '--n_files', type=int, default=0, help='number of files to use.')
    parser.add_argument('--batch_size', type=int, default=64, help='minibatch size')
    parser.add_argument('--tags', default='splicing', help='tags for comet-ml experiment')
    parser.add_argument('--lr', type=float, default=0.0)
    parser.add_argument('--device', type=int, default=-1, help='compute device to use')
    parser.add_argument('--test', action='store_true')
    parser.set_defaults(test=False)
    args = parser.parse_args()
    main(args)