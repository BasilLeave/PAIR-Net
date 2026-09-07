import numpy as np
import os
import shutil
import argparse

# import models
# import wandb

_DATASET = ('UCMerced', 'MLRSNet', 'AID')
_SCHEMES = ('BCE', 'ELR', 'SAT', 'LCR', 'JoCoR', 'RCML')

AID_CATEGORY = ['airplane', 'bare soil', 'buildings', 'cars', 'chaparral', 'court', 'dock', 'field', 'grass',
                'mobile home', 'pavement', 'sand', 'sea', 'ship', 'tanks', 'trees', 'water']
UCMERCED_CATEGORY = ['airplane', 'bare soil', 'buildings', 'cars', 'chaparral', 'court', 'dock', 'field', 'grass',
                     'mobile home', 'pavement', 'sand', 'sea', 'ship', 'tanks', 'trees', 'water']
MLRSNET_CATEGORY = ['airplane','airport','bare soil','baseball diamond','basketball court','beach','bridge','buildings',
                    'cars','chaparral','cloud','containers','crosswalk','dense residential area','desert','dock','factory',
                    'field','football field','forest','freeway','golf course','grass','greenhouse','gully','habor','intersection',
                    'island','lake','mobile home','mountain','overpass','park','parking lot','parkway','pavement','railway',
                    'railway station','river','road','roundabout','runway','sand','sea','ships','snow','snowberg',
                    'sparse residential area','stadium','swimming pool','tanks','tennis court','terrace','track','trail',
                    'transmission tower','trees','water','wetland','wind turbine']

_DATASET = ('UCMerced', 'MLRSNet', 'AID')
_SCHEMES = ('BCE', 'ELR', 'SAT', 'LCR', 'JoCoR', 'RCML')


parser = argparse.ArgumentParser(description='Training Settings')
parser.add_argument('--noise_rate', type=float, default=40, help='Noise rate percentage')
parser.add_argument('--noise_type', type=str, default='mix', choices=['additive', 'subtractive', 'mix'], help='Type of label noise')
parser.add_argument('--img_size', type=int, default=448, help='Input image size')
parser.add_argument('--dataset', type=str, default='UCMerced', help='Dataset name', choices=_DATASET)
parser.add_argument('--seed', type=int, default=2, help='Random seed for reproducibility')
parser.add_argument('--save_path', type=str, default=None, help='Path to save results')
parser.add_argument('--bsize', type=int, default=16, help='Batch size for training')
parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')

parser.add_argument('--backbone', default='resnet50', type=str,
                    help="Name of the convolutional backbone to use")
parser.add_argument('--scheme', type=str, default='RCML', help='Loss scheme', choices=_SCHEMES)
parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')

parser.add_argument('--lr_mult', type=float, default=10)
parser.add_argument('--lam1', type=float, default=1)
parser.add_argument('--lam2', type=float, default=0.1)
parser.add_argument('--lam3', type=float, default=0.6)
parser.add_argument('--Es', type=int, default=5)
parser.add_argument('--Tk', type=int, default=50)
parser.add_argument('--tau', type=float, default=0.05)
parser.add_argument('--swap_rate', type=float, default=0.9)
parser.add_argument('--alpha', type=float, default=0.6)

parser.add_argument('--gpu', type=str, default='1', help='the used gpu id')
args = parser.parse_args()
print(args)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
import os
from instrumentation import compute_metrics
import losses
import datasets
import random
import datetime
import matplotlib.pyplot as plt

from models import ImageClassifier_feats

torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.device = device


def format_duration(seconds):
    seconds = max(int(seconds), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{secs:02d}'

def run_train(args):
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    if args.dataset == 'AID':
        category = AID_CATEGORY
        num_class = 17
        num_train = 2400
        path_to_dataset = 'data/AID_multilabel'
    elif args.dataset == 'MLRSNet':
        category = MLRSNET_CATEGORY
        num_class = 60
        num_train = 87319
        path_to_dataset = 'data/MLRSNet'
        args.bsize = 32
    elif args.dataset == 'UCMerced':
        category = UCMERCED_CATEGORY
        num_class = 17
        num_train = 1680
        path_to_dataset = 'data/UCMerced_LandUse'
    
    args.classes = category
    args.num_class = num_class
    args.num_train = num_train
    args.path_to_dataset = path_to_dataset

    save_suffix = './results'

    # set save dir and logger
    if not args.save_path:
        args.save_path = (f"{save_suffix}/{args.scheme}/{args.dataset}/{args.noise_type}_{args.noise_rate}/{datetime.datetime.now()}").replace(
            ' ', '-')
    os.makedirs(args.save_path, exist_ok=True)

    logger_file = os.path.join(args.save_path, 'logger.txt')
    # log the results
    with open(logger_file, 'w') as logger:
        logger.write(f'Epoch\tmAP\tHamming\n')

    args.logger_file = logger_file

    dataset = datasets.get_dataset(args)


    if args.noise_rate != 0:
        if args.noise_type == 'additive':
            dataset['train'].inject_additive_noise(args)

        elif args.noise_type == 'subtractive':
            dataset['train'].inject_subtractive_noise(args)

        elif args.noise_type == 'mix':
            dataset['train'].inject_mix_noise(args)

    # save args
    with open(os.path.join(args.save_path, 'args.txt'), 'w') as f:
        for arg in vars(args):
            arg = str(arg)
            f.write(f'{arg}: {getattr(args, arg)}\n')

    dataloader = {}
    for phase in ['train', 'test']:
        dataloader[phase] = torch.utils.data.DataLoader(
            dataset[phase],
            batch_size=args.bsize,
            shuffle=phase == 'train',
            sampler=None,
            num_workers=args.num_workers,
            drop_last=False,
            pin_memory=True
        )

    model1 = ImageClassifier_feats(args.num_class)
    model2 = ImageClassifier_feats(args.num_class)


    optimizer1 = torch.optim.Adam(model1.parameters(), lr=args.lr)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=args.lr)

    total_iterations = len(dataloader['train']) * args.num_epochs

    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=total_iterations)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=total_iterations)

    device = f'cuda' if torch.cuda.is_available() else 'cpu'
    model1.to(device)
    model2.to(device)
    args.device = device

    criterion = losses.get_criterion(args)

    best_mAP = .0
    start_epoch = 0

    max_epoch = args.num_epochs
    num_iterations = len(dataloader['train'])
    train_start_time = datetime.datetime.now().timestamp()
    for epoch in range(start_epoch, max_epoch):
        print(f'Epoch {epoch + 1}/{max_epoch}')


        model1.train()
        model2.train()

        with torch.set_grad_enabled(True):
            for iteration, (image, label, index) in enumerate(dataloader['train']):
                # Move data to GPU
                image = image.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)

                logits1, intermediate_feats1 = model1(image)
                logits2, intermediate_feats2 = model2(image)
                if logits1.dim() == 1:
                    logits1 = torch.unsqueeze(logits1, 0)
                    logits2 = torch.unsqueeze(logits2, 0)

                loss1, loss2 = criterion(logits1, logits2, intermediate_feats1, intermediate_feats2, label)

                optimizer1.zero_grad()
                loss1.backward()
                optimizer1.step()
                optimizer2.zero_grad()
                loss2.backward()
                optimizer2.step()
                scheduler1.step()
                scheduler2.step()

                completed_iterations = epoch * num_iterations + iteration + 1
                elapsed_time = datetime.datetime.now().timestamp() - train_start_time
                avg_iter_time = elapsed_time / max(completed_iterations, 1)
                remaining_iterations = max(total_iterations - completed_iterations, 0)
                remaining_time = avg_iter_time * remaining_iterations

                print(
                    f'Epoch {epoch + 1}/{max_epoch}, '
                    f'Iteration {iteration + 1}/{num_iterations}, '
                    f'Loss1: {loss1.item()}, '
                    f'Loss2: {loss2.item()}, '
                    f'Elapsed: {format_duration(elapsed_time)}, '
                    f'ETA: {format_duration(remaining_time)}'
                )
            

        criterion.end_of_epoch()

        # evaluation
        if epoch % 10 == 0 or epoch == max_epoch - 1: 
            test_len = len(dataset['test'])
            mAP1, hamming1 = validate(dataloader['test'], test_len, model1, epoch, args)
            mAP2, hamming2 = validate(dataloader['test'], test_len, model2, epoch, args)

            mAP = .0
            haming = .0
            if mAP1 > mAP2:
                mAP = mAP1
                hamming = hamming1
                model = model1
            else:
                mAP = mAP2
                hamming = hamming2
                model = model2

            # wandb.log({"mAP": mAP, "hamming": hamming})
            is_best = mAP > best_mAP
            best_mAP = max(mAP, best_mAP)
            if is_best:
                best_hamming = hamming

            print(f"Epoch {epoch}: test mAP macro {mAP:.3f}, hamming loss {hamming:.3f}")

            # log the results
            with open(args.logger_file, 'a') as logger:
                logger.write(f'{epoch}\t{mAP:.3f}\t{hamming:.3f}\n')


    print('Training procedure completed!')


def validate(val_loader, test_len, model, epoch, args):
    y_pred = np.zeros((test_len, args.num_class))
    y_true = np.zeros((test_len, args.num_class))
    batch_stack = 0

    model = model.eval()
    with torch.set_grad_enabled(False):
        for image, label, index in val_loader:
            # Move data to GPU
            image = image.to(args.device, non_blocking=True)
            label = label.to(args.device, non_blocking=True)

            logits, _ = model(image)
            if logits.dim() == 1:
                logits = torch.unsqueeze(logits, 0)

            preds = torch.sigmoid(logits)

            preds_np = preds.cpu().numpy()
            this_batch_size = preds_np.shape[0]
            y_pred[batch_stack: batch_stack + this_batch_size] = preds_np
            y_true[batch_stack: batch_stack + this_batch_size] = label.cpu().numpy()
            batch_stack += this_batch_size

    metrics = compute_metrics(y_pred, y_true)
    map = metrics['map']
    hamming = metrics['hamming']

    return map, hamming


if __name__ == '__main__':
    run_train(args)
