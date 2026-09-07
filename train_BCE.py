import numpy as np
import os

import argparse
import torch
import os
from instrumentation import compute_metrics
import losses
import datasets
import random
import datetime
from models import ImageClassifier

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

class Config:
    pass

parser = argparse.ArgumentParser(description='Training Settings')
parser.add_argument('--noise_rate', type=float, default=40, help='Noise rate percentage')
parser.add_argument('--noise_type', type=str, default='mix', choices=['additive', 'subtractive', 'mix'], help='Type of label noise')
parser.add_argument('--img_size', type=int, default=448, help='Input image size')
parser.add_argument('--dataset', type=str, default='UCMerced', help='Dataset name', choices=_DATASET)
parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
parser.add_argument('--save_path', type=str, default=None, help='Path to save results')
parser.add_argument('--bsize', type=int, default=64, help='Batch size for training')
parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')

parser.add_argument('--backbone', default='resnet50', type=str,
                    help="Name of the convolutional backbone to use")
parser.add_argument('--scheme', type=str, default='BCE', help='Loss scheme', choices=_SCHEMES)
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

parser.add_argument('--gpu', type=str, default='0', help='the used gpu id')

args = parser.parse_args()
print(args)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True
args.device = device

def run_train(args):
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    if args.dataset == 'AID':
        category = AID_CATEGORY
        num_class = 17
        num_train = 2400
        args.bsize = 16
        path_to_dataset = 'data/AID_multilabel'
    elif args.dataset == 'MLRSNet':
        category = MLRSNET_CATEGORY
        num_class = 60
        num_train = 87319
        path_to_dataset = 'data/MLRSNet'
        args.bsize = 64
    elif args.dataset == 'UCMerced':
        category = UCMERCED_CATEGORY
        num_class = 17
        num_train = 1680
        args.bsize = 16
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
            # dataset['train'].inject_noise(args)

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

    model = ImageClassifier(args.num_class)

    max_epoch = args.num_epochs
    num_iterations = len(dataloader['train'])
    print(f'Number of iterations: {num_iterations}')
    total_iterations = max_epoch * num_iterations

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iterations)

    model.to(device)

    criterion = losses.get_criterion(args)

    best_mAP = .0
    start_epoch = 0

    for epoch in range(start_epoch, max_epoch):
        print(f'Epoch {epoch + 1}/{max_epoch}')
        model.train()
        with torch.set_grad_enabled(True):
            for iteration, (image, label, index) in enumerate(dataloader['train']):
                # Move data to GPU
                image = image.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)

                optimizer.zero_grad()
                logits = model(image)
                if logits.dim() == 1:
                    logits = torch.unsqueeze(logits, 0)

                loss = criterion(logits, label, index)
                loss.backward()
                optimizer.step()
                scheduler.step()

                # log the loss
                print(f'Epoch {epoch + 1}/{max_epoch}, Iteration {iteration + 1}/{num_iterations}, Loss: {loss.item()}')

        criterion.end_of_epoch()

        # evaluation 
        if epoch % 10 == 0 or epoch == max_epoch - 1:
            test_len = len(dataset['test'])
            mAP, hamming = validate(dataloader['test'], test_len, model, epoch, args)
            print(f"Epoch {epoch}: test mAP macro {mAP:.3f}, hamming loss {hamming:.3f}")

            # remember best mAP
            is_best = mAP > best_mAP
            best_mAP = max(mAP, best_mAP)

            if is_best:
                best_hamming = hamming


    with open(logger_file, 'a') as logger:
        logger.write(f'Best mAP: {best_mAP}\n')
        logger.write(f'Best Hamming: {best_hamming}\n')


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

            logits = model(image)
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
    print(f"{epoch}: test mAP macro {map:.3f}, hamming loss {hamming:.3f}")

    # log the results
    with open(args.logger_file, 'a') as logger:
        logger.write(f'{epoch}\t{map:.3f}\t{hamming:.3f}\n')
    return map, hamming


if __name__ == '__main__':
    run_train(args)







