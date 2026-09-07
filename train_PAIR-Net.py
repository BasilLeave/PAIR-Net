import os
import numpy as np
import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


AID_CATEGORY = ['airplane', 'bare soil', 'buildings', 'cars', 'chaparral', 'court', 'dock', 'field', 'grass',
                'mobile home', 'pavement', 'sand', 'sea', 'ship', 'tanks', 'trees', 'water']
UCMERCED_CATEGORY = ['airplane', 'bare soil', 'buildings', 'cars', 'chaparral', 'court', 'dock', 'field', 'grass',
                     'mobile home', 'pavement', 'sand', 'sea', 'ship', 'tanks', 'trees', 'water']


parser = argparse.ArgumentParser(description='Training Settings')
parser.add_argument('--noise_rate', type=float, default=40, help='Noise rate percentage')
parser.add_argument('--noise_type', type=str, default='subtractive', choices=['additive', 'subtractive', 'mix'], help='Type of label noise')
parser.add_argument('--img_size', type=int, default=448, help='Input image size')
parser.add_argument('--num_class', type=int, default=17, help='Number of classes')
parser.add_argument('--dataset', type=str, default='UCMerced', help='Dataset name')
parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
parser.add_argument('--save_path', type=str, default=None, help='Path to save results')
parser.add_argument('--bsize', type=int, default=16, help='Batch size for training')
parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
parser.add_argument('--backbone', default='resnet50', type=str,
                    help="Name of the convolutional backbone to use")
parser.add_argument('--scheme', type=str, default='BCE', help='Loss scheme')
parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')

# learnable
parser.add_argument('--learnable_label_embed', type=str2bool, nargs='?', const=True, default=True,
                    help='Whether to use learnable label embeddings')
parser.add_argument('--no_learnable_label_embed', dest='learnable_label_embed', action='store_false')
parser.add_argument('--random_init', type=str2bool, nargs='?', const=True, default=False,
                    help='Whether to use random initialization for label embeddings')
parser.add_argument('--no_random_init', dest='random_init', action='store_false')
parser.add_argument('--clip_name', type=str, default='ViT-L/14', choices=['ViT-L/14', 'ViT-B/32', 'ViT-B/16'], help='CLIP model name')
parser.add_argument('--n_ctx', type=int, default=16, help='Number of context tokens')
parser.add_argument('--class_token_position', type=str, default='end', choices=['front', 'middle', 'end'], help='Position of class token in prompt')


# multi-head attention
parser.add_argument('--enc_layers', default=0, type=int,
                        help="Number of encoding layers in the transformer")
parser.add_argument('--dec_layers', default=1, type=int,
                    help="Number of decoding layers in the transformer")
parser.add_argument('--dropout', default=0.1, type=float,
                    help="Dropout applied in the transformer")
parser.add_argument('--nheads', default=4, type=int,
                    help="Number of attention heads inside the transformer's attentions")
parser.add_argument('--pre_norm', action='store_true')
parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine'),
                    help="Type of positional embedding to use on top of the image features")
parser.add_argument('--keep_other_self_attn_dec', default=False,
                    help='keep the other self attention modules in transformer decoders, which will be removed default.')
parser.add_argument('--keep_first_self_attn_dec', default=False,
                    help='keep the first self attention module in transformer decoders, which will be removed default.')
parser.add_argument('--pretrained', action='store_true', default=True)
parser.add_argument('--get_cam', action='store_true', default=True)

# filter
parser.add_argument('--filter_start_epoch', default=2, type=int,
                    help='Start epoch index for noisy-label filtering.')
parser.add_argument('--filter_epochs', default=8, type=int)
parser.add_argument('--filter_threshold', default=0.2, type=float,
                    help='Filter threshold for noisy-label detection.')
parser.add_argument('--gpu', type=str, default='2', help='GPU device ID to use')

parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--lr_plateau_factor', type=float, default=0.8,
                    help='Factor used by ReduceLROnPlateau when mAP stops improving.')
parser.add_argument('--lr_plateau_patience', type=int, default=1,
                    help='Number of validation intervals with no mAP improvement before reducing lr.')
parser.add_argument('--min_lr_ratio', type=float, default=0.05,
                    help='Minimum learning-rate ratio relative to the initial lr for plateau scheduling.')
parser.add_argument('--eval_interval', type=int, default=1,
                    help='Evaluate mAP every N epochs. Default 1 lets the scheduler react without 5-epoch delay.')
parser.add_argument('--scheduler_start_after_filter', type=str2bool, nargs='?', const=True, default=True,
                    help='Start mAP-driven LR scheduling only after the hard-filter window ends.')
parser.add_argument('--protect_on_lr_reduce', type=str2bool, nargs='?', const=True, default=True,
                    help='When LR first drops, restore online model to best EMA/eval state and enter protected training.')
parser.add_argument('--protect_lr_ratio', type=float, default=0.1,
                    help='Protected online LR ratio relative to the initial LR after EMA protection is triggered.')
parser.add_argument('--freeze_ema_after_protect', type=str2bool, nargs='?', const=True, default=False,
                    help='Freeze EMA updates after protection so EMA remains the stable anchor.')
parser.add_argument('--adaptive_ema_after_protect', type=str2bool, nargs='?', const=True, default=True,
                    help='After protection, keep EMA updating but slow it according to the reduced online LR.')
parser.add_argument('--eval_online_after_protect', type=str2bool, nargs='?', const=True, default=False,
                    help='After protection, evaluate the online model because it is the final model being refined.')
parser.add_argument('--final_use_online_after_protect', type=str2bool, nargs='?', const=True, default=False,
                    help='Keep the protected online model as final export instead of overwriting it with frozen EMA.')
parser.add_argument('--use_ema', type=str2bool, nargs='?', const=True, default=True,
                    help='Use EMA weights to stabilize late-epoch performance.')
parser.add_argument('--ema_decay', default=0.9997, type=float,
                    help='EMA decay factor.')
parser.add_argument('--ema_start_epoch', default=0, type=int,
                    help='Start epoch index for EMA updates.')
parser.add_argument('--eval_with_ema', type=str2bool, nargs='?', const=True, default=True,
                    help='Evaluate with EMA model when available.')
parser.add_argument('--final_use_ema', type=str2bool, nargs='?', const=True, default=True,
                    help='Export final model using EMA weights when available.')
parser.add_argument('--load_best_at_end', type=str2bool, nargs='?', const=True, default=False,
                    help='Load best checkpoint at the end.')
parser.add_argument('--save_best_checkpoint', type=str2bool, nargs='?', const=True, default=False,
                    help='Save best checkpoint during training.')


args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


import torch
from model import build_l2v
from instrumentation import compute_metrics
import losses
import datasets
import random
from utils import visualization

import datetime
import time

import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True
args.device = device

def build_ema_model(args, source_model, device):
    # Avoid deepcopy(module): some parametrized/weight-norm tensors are not deepcopy-safe.
    ema_model = build_l2v(args)
    ema_model.load_state_dict(source_model.state_dict(), strict=True)
    ema_model.to(device)
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)
    return ema_model

def update_ema_model(model, ema_model, decay):
    with torch.no_grad():
        model_state = model.state_dict()
        ema_state = ema_model.state_dict()
        for name, ema_value in ema_state.items():
            model_value = model_state[name].detach()
            if torch.is_floating_point(ema_value):
                ema_value.mul_(decay).add_(model_value, alpha=(1.0 - decay))
            else:
                ema_value.copy_(model_value)

def get_current_ema_decay(args, optimizer, ema_protected):
    if not (ema_protected and args.adaptive_ema_after_protect):
        return args.ema_decay

    current_lr = optimizer.param_groups[0]['lr']
    lr_ratio = max(0.0, min(1.0, current_lr / max(args.lr, 1e-12)))
    ema_alpha = (1.0 - args.ema_decay) * lr_ratio
    adaptive_decay = 1.0 - ema_alpha
    return min(0.999999, max(args.ema_decay, adaptive_decay))

def clone_state_dict_to_cpu(model):
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }

def build_plateau_scheduler(args, optimizer, min_lr):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=args.lr_plateau_factor,
        patience=args.lr_plateau_patience,
        threshold=1e-4,
        threshold_mode='rel',
        cooldown=0,
        min_lr=min_lr,
    )

def format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'

def run_train(args):
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)      

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        device = torch.device('cuda')
        args.device = device
        print("Using CUDA!")
    else:
        device = torch.device('cpu')
        args.device = device
        print("Using CPU!")

    if args.dataset == 'AID':
        category = AID_CATEGORY
        args.path_to_dataset = 'data/AID_multilabel'
        args.cam_to_dataset = 'data/AID_multilabel/CAMs/'

    elif args.dataset == 'UCMerced':
        category = UCMERCED_CATEGORY
        args.path_to_dataset = 'data/UCMerced_LandUse'
        args.cam_to_dataset = 'data/UCMerced_LandUse/CAMs/'

    args.classes = category

    if args.clip_name == 'ViT-L/14':
        args.hidden_dim = 768
    elif args.clip_name == 'ViT-B/32':
        args.hidden_dim = 512
    elif args.clip_name == 'ViT-B/16':
        args.hidden_dim = 512
    
    args.dim_feedforward = args.hidden_dim * 4
    print(args)

    save_suffix = './results'

    # set save dir and logger
    if not args.save_path:
        args.save_path = (f"{save_suffix}/PAIR-Net/{args.dataset}/{args.noise_type}_{args.noise_rate}/{datetime.datetime.now()}").replace(
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
            true_positive_label_matrix = dataset['train'].true_positive_label_matrix
            false_positive_label_matrix = dataset['train'].false_positive_label_matrix

        elif args.noise_type == 'subtractive':
            dataset['train'].inject_subtractive_noise(args)

        elif args.noise_type == 'mix':
            dataset['train'].inject_mix_noise(args)
            true_positive_label_matrix = dataset['train'].true_positive_label_matrix
            false_positive_label_matrix = dataset['train'].false_positive_label_matrix

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


    model = build_l2v(args)

    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'prompt_learner' in name:
                print(name)


    max_epoch = args.num_epochs
    num_iterations = len(dataloader['train'])
    print(f'Number of iterations: {num_iterations}')
    total_iterations = max_epoch * num_iterations

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    min_lr = args.lr * args.min_lr_ratio
    scheduler = build_plateau_scheduler(args, optimizer, min_lr)

    model.to(device)
    ema_model = build_ema_model(args, model, device) if args.use_ema else None

    criterion = losses.get_criterion(args)

    best_mAP = .0
    best_hamming = None
    best_epoch = -1
    epochs_without_improvement = 0
    best_eval_state = None
    best_model_path = os.path.join(args.save_path, 'model_best.pth')
    best_ckpt_saved = False
    start_epoch = 0

    # warmup epoch
    args.method = 'refineCAMs'
    true_sum = 0
    false_sum = 0
    filter_end_epoch = args.filter_start_epoch + args.filter_epochs
    scheduler_start_epoch = filter_end_epoch if args.scheduler_start_after_filter else start_epoch
    ema_started = False
    ema_update_frozen = False
    ema_protected = False
    protect_epoch = -1
    train_start_time = time.time()

    for epoch in range(start_epoch, max_epoch):
        epoch_start_time = time.time()
        print(f'Epoch {epoch + 1}/{max_epoch}')
        model.train()
        with torch.set_grad_enabled(True):
            for iteration, (image, label, index, cams) in enumerate(dataloader['train']):
                # Move data to GPU
                image = image.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)

                optimizer.zero_grad()
                logits, _ = model(image)

                loss = criterion(logits, label, index)

                loss.backward()
                optimizer.step()

                if args.use_ema and epoch >= args.ema_start_epoch and not ema_update_frozen:
                    if not ema_started:
                        ema_model.load_state_dict(model.state_dict(), strict=True)
                        ema_started = True
                    else:
                        current_ema_decay = get_current_ema_decay(args, optimizer, ema_protected)
                        update_ema_model(model, ema_model, current_ema_decay)

                # log the loss
                completed_iterations = epoch * num_iterations + iteration + 1
                elapsed_time = time.time() - train_start_time
                avg_iter_time = elapsed_time / max(completed_iterations, 1)
                remaining_iterations = max(total_iterations - completed_iterations, 0)
                remaining_time = avg_iter_time * remaining_iterations
                print(
                    f'Epoch {epoch + 1}/{max_epoch}, '
                    f'Iteration {iteration + 1}/{num_iterations}, '
                    f'Loss: {loss.item()}, '
                    f'Elapsed: {format_duration(elapsed_time)}, '
                    f'ETA: {format_duration(remaining_time)}'
                )

        criterion.end_of_epoch()


        in_filter_window = args.filter_start_epoch <= epoch < filter_end_epoch and args.noise_type != 'subtractive' and args.noise_rate > 0
        current_noisy_mask = np.zeros_like(dataset['train'].label_matrix, dtype=np.int8)
        epoch_true_sum = 0
        epoch_false_sum = 0
        if in_filter_window:
            with torch.no_grad():
                current_noisy_mask = filter_noisy_labels_by_containment(
                    model,
                    dataloader['train'],
                    dataset['train'],
                    args,
                    epoch=epoch,
                    filter_start_epoch=args.filter_start_epoch,
                    filter_end_epoch=filter_end_epoch,
                )
                true_prediction_matrix = current_noisy_mask.astype(bool) & false_positive_label_matrix.astype(bool)
                false_prediction_matrix = current_noisy_mask.astype(bool) & true_positive_label_matrix.astype(bool)
                epoch_true_sum = int(true_prediction_matrix.sum())
                epoch_false_sum = int(false_prediction_matrix.sum())
                print(f'true_prediction_sum: {epoch_true_sum}')
                print(f'false_prediction_sum: {epoch_false_sum}')
                print('true_prediction_rate:', epoch_true_sum * 1.0 / false_positive_label_matrix.sum())
                print('false_prediction_rate:', epoch_false_sum * 1.0 / true_positive_label_matrix.sum())

                with open(logger_file, 'a') as logger:
                    logger.write(f'true_prediction_sum: {epoch_true_sum}\n')
                    logger.write(f'false_prediction_sum: {epoch_false_sum}\n')
                    logger.write(
                        f'true_prediction_rate: {epoch_true_sum * 1.0 / false_positive_label_matrix.sum()}\n')
                    logger.write(
                        f'false_prediction_rate: {epoch_false_sum * 1.0 / true_positive_label_matrix.sum()}\n')


            dataset['train'].remove_noisy_labels(current_noisy_mask)
            true_sum += epoch_true_sum
            false_sum += epoch_false_sum
                 
        # evaluation
        eval_interval = max(1, args.eval_interval)
        if epoch % eval_interval == 0 or epoch == max_epoch - 1:
            test_len = len(dataset['test'])
            eval_model = model
            eval_uses_ema = False
            if args.use_ema and args.eval_with_ema and ema_started:
                if not (args.eval_online_after_protect and ema_protected):
                    eval_model = ema_model
                    eval_uses_ema = True
            mAP, hamming = validate(dataloader['test'], test_len, eval_model, epoch, args)
            print(f"Epoch {epoch}: test mAP macro {mAP:.3f}, hamming loss {hamming:.3f}")
            prev_lr = optimizer.param_groups[0]['lr']
            scheduler_active = epoch >= scheduler_start_epoch
            if scheduler_active:
                scheduler.step(mAP)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != prev_lr:
                if args.protect_on_lr_reduce and args.use_ema and not ema_protected:
                    if best_eval_state is None:
                        print('Warning: LR protection skipped because no best EMA/eval state is available yet.')
                    else:
                        protected_lr = max(args.lr * args.protect_lr_ratio, min_lr)
                        model.load_state_dict(best_eval_state, strict=True)
                        if ema_model is not None:
                            ema_model.load_state_dict(best_eval_state, strict=True)
                            ema_model.eval()
                        if args.freeze_ema_after_protect:
                            ema_update_frozen = True
                        ema_protected = True
                        protect_epoch = epoch

                        # Reset Adam momentum after jumping online weights to the EMA anchor.
                        optimizer = torch.optim.Adam(model.parameters(), lr=protected_lr)
                        scheduler = build_plateau_scheduler(args, optimizer, min_lr)

            # # remember best mAP and save checkpoint
            is_best = mAP > best_mAP
            if is_best:
                best_mAP = mAP
                best_hamming = hamming
                best_epoch = epoch
                best_eval_state = clone_state_dict_to_cpu(eval_model)
                epochs_without_improvement = 0
                if args.save_best_checkpoint:
                    try:
                        torch.save(model.state_dict(), best_model_path)
                        best_ckpt_saved = True
                    except Exception as e:
                        best_ckpt_saved = False
                        print(f'Warning: failed to save best checkpoint: {e}')
                        with open(logger_file, 'a') as logger:
                            logger.write(f'Warning: failed to save best checkpoint: {e}\n')
                print(f'New best model at epoch {epoch}: mAP={best_mAP:.3f}, hamming={best_hamming:.3f}')
            else:
                epochs_without_improvement += 1

        epoch_elapsed_time = time.time() - epoch_start_time
        total_elapsed_time = time.time() - train_start_time
        completed_epochs = epoch - start_epoch + 1
        avg_epoch_time = total_elapsed_time / max(completed_epochs, 1)
        remaining_epochs = max(max_epoch - epoch - 1, 0)
        epoch_based_remaining_time = avg_epoch_time * remaining_epochs
        time_msg = (
            f'Epoch {epoch + 1} finished. '
            f'Epoch time: {format_duration(epoch_elapsed_time)}, '
            f'Total elapsed: {format_duration(total_elapsed_time)}, '
            f'Epoch-based ETA: {format_duration(epoch_based_remaining_time)}'
        )
        print(time_msg)
        with open(logger_file, 'a') as logger:
            logger.write(f'{time_msg}\n')

    if args.load_best_at_end and best_ckpt_saved and os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f'Loaded best checkpoint from epoch {best_epoch} for final export.')
        with open(logger_file, 'a') as logger:
            logger.write(f'Loaded best checkpoint: {best_model_path}\n')
    elif args.final_use_online_after_protect and ema_protected:
        print('Kept protected online model as final model export.')
        with open(logger_file, 'a') as logger:
            logger.write('Kept protected online model as final model export.\n')
    elif args.final_use_ema and args.use_ema and ema_started:
        model.load_state_dict(ema_model.state_dict(), strict=True)
        print('Loaded EMA checkpoint as final model export.')
        with open(logger_file, 'a') as logger:
            logger.write('Loaded EMA checkpoint as final model export.\n')


    with open(logger_file, 'a') as logger:
        logger.write(f'Best epoch: {best_epoch}\n')
        logger.write(f'Best mAP: {best_mAP}\n')
        logger.write(f'Best hamming: {best_hamming}\n')


    print('Training procedure completed!')


def validate(val_loader, test_len, model, epoch, args):
    y_pred = np.zeros((test_len, args.num_class))
    y_true = np.zeros((test_len, args.num_class))
    batch_stack = 0

    model = model.eval()
    with torch.set_grad_enabled(False):
        for image, label, index, _ in val_loader:
            # Move data to GPU
            image = image.to(args.device, non_blocking=True)
            label = label.to(args.device, non_blocking=True)

            logits, _ = model(image)
            if logits.dim() == 1:
                logits = torch.unsqueeze(logits, 0)

            preds = torch.sigmoid(logits)
            preds = torch.clamp(preds, 0, 1)

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




def refine_cam_with_cross_attn(cams, cross_attn_list):
    """
        cams: original CAMs from the model
        cross_attn_list: cross-attention weights from the transformer decoder
    """
    B, C, H_cam, W_cam = cams.shape

    if len(cross_attn_list) > 1:
        attn_weight = torch.stack(cross_attn_list, dim=0)[:-1]
        attn_weight = torch.mean(attn_weight, dim=0)
    else:
        attn_weight = cross_attn_list[0]

    h_attn = int(attn_weight.shape[-1] ** 0.5)
    w_attn = h_attn
    attn_map = attn_weight.reshape(B, C, h_attn, w_attn)

    attn_map_resized = F.interpolate(attn_map, size=(H_cam, W_cam), mode='bilinear', align_corners=False)

    def normalize_cam(x):
        x_min = x.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        x_max = x.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        return (x - x_min) / (x_max - x_min + 1e-8)

    norm_attn = normalize_cam(attn_map_resized)
    norm_attn = norm_attn.to(cams.device)
    refined_cam = cams * norm_attn
    refined_cam = normalize_cam(refined_cam)

    return refined_cam

def filter_noisy_labels_by_containment(
    model,
    train_loader,
    train_dataset,
    args,
    epoch=None,
    filter_start_epoch=None,
    filter_end_epoch=None,
):

    noisy_labels = np.zeros_like(train_dataset.label_matrix, dtype=np.int8)
    model.eval()

    HIGH_CONNECTIVITY_CLASSES = [
        'pavement', 'grass', 'buildings', 'bare soil', 'trees',
        'cars', 'water', 'sand', 'sea'
    ]
    class_name_to_idx = {name: i for i, name in enumerate(args.classes)}
    high_conn_indices = [class_name_to_idx[name] for name in HIGH_CONNECTIVITY_CLASSES if name in class_name_to_idx]

    # load filtering parameters
    Filter_threshold = float(args.filter_threshold)  

    Protection_Set = {
        'dock': {'ship', 'water'},
        'ship': {'dock', 'water'},

        'airplane': {'pavement'},

        'chaparral': {
            'field', 'sand', 'tanks',
            'pavement', 'grass', 'trees', 'bare soil',
            'water',
        },

        'court': {
            'buildings', 'pavement', 'bare soil',
            'field', 'grass', 'trees',
            'water',
        },

        'field': {'trees', 'water', 'sand', 'buildings', 'pavement'},
        'tanks': {'pavement', 'bare soil', 'buildings'},
    }
    FAST_FILTER_CLASSES = {'dock', 'sea', 'ship', 'tanks', 'court', 'chaparral', 'airplane'}
    

    exclude_idx_map = {}
    for cls_j_name, exclude_names in Protection_Set.items():
        if cls_j_name not in class_name_to_idx:
            continue
        j_idx = class_name_to_idx[cls_j_name]
        exclude_idx_map[j_idx] = {
            class_name_to_idx[n]
            for n in exclude_names
            if n in class_name_to_idx
        }

    total_filtered_count = 0
    false_positive_matrix = train_dataset.false_positive_label_matrix.astype(np.int8)

    with torch.no_grad():
        for x, y, index, cams in train_loader:
            x = x.to(args.device)
            N, C, H, W = cams.size()
            if args.method == 'refineCAMs':
                logits, attn_maps = model(x)
                cams = refine_cam_with_cross_attn(cams, attn_maps)
            elif args.method == 'attn':
                logits, attn_maps = model(x)
                attn_weight = torch.stack(attn_maps, dim=0)[:-1]
                attn_weight = torch.mean(attn_weight, dim=0)
                h_attn = int(attn_weight.shape[-1] ** 0.5)
                w_attn = h_attn
                cams = attn_weight.reshape(N, C, h_attn, w_attn)
                def normalize_cam(x):
                    x_min = x.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
                    x_max = x.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
                    return (x - x_min) / (x_max - x_min + 1e-8)
                cams = normalize_cam(cams)
            else:
                logits, _ = model(x)

            cams_np = cams.cpu().numpy()
            index_np = index.cpu().numpy()
            y_np = y.cpu().numpy()
            scores_np = torch.sigmoid(logits).cpu().numpy()

            for i in range(N):
                data_idx = index_np[i]
                positive_indices = np.where(y_np[i] == 1)[0]

                if len(positive_indices) < 2:
                    continue

                # generate binary masks for all positive classes in the current sample
                masks = {
                    j: visualization.get_binary_mask(cams_np[i, j]).astype(bool)
                    for j in positive_indices
                }

                for j in positive_indices:
                    cls_j = args.classes[j]
                    if cls_j in FAST_FILTER_CLASSES and epoch != filter_start_epoch:
                        continue

                    if j in high_conn_indices:
                        continue
                    
                    mask_j = masks[j]
                    area_j = np.sum(mask_j)
                    if area_j == 0:
                        continue

                    # U_j = ∪_{k != j} M_k
                    union_others = np.zeros_like(mask_j, dtype=bool)
                    exclude_k_set = exclude_idx_map.get(j, set())
                    contributors = []
                    max_foreground_contrib = 0.0
                    for k in positive_indices:
                        if j == k:
                            continue
                        if k in exclude_k_set:
                            continue

                        if scores_np[i, j] > (scores_np[i, k] - 0.15):
                            continue

                        contrib_jk = np.sum(mask_j & masks[k]) / (area_j + 1e-8)
                        if contrib_jk <= 0:
                            continue
                        contributors.append((args.classes[k], contrib_jk))
                        if k not in high_conn_indices:
                            max_foreground_contrib = max(max_foreground_contrib, contrib_jk)
                        
                        union_others |= masks[k]

                    # SCS(j) = |M_j ∩ U_j| / (|M_j| + eps)
                    overlap_with_others = np.sum(mask_j & union_others)
                    SCS_j = overlap_with_others / (area_j + 1e-8)

                    is_noise = SCS_j >= Filter_threshold
                    if is_noise:
                        label_status = 'noise' if false_positive_matrix[data_idx, j] == 1 else 'clean'
                        top_contributors = sorted(contributors, key=lambda x: x[1], reverse=True)[:3]
                        contributor_msg = ','.join(
                            f'{cls_name}:{score:.4f}'
                            for cls_name, score in top_contributors
                        )
                        msg = (
                            f'idx={data_idx}, class={args.classes[j]}, '
                            f'label_status={label_status}, '
                            f'SCS={SCS_j:.4f}, '
                            f'contributors={contributor_msg}'
                        )
                        print(msg)
                        with open(args.logger_file, 'a') as logger:
                                logger.write(f'{msg}\n')
                                
                        noisy_labels[data_idx, j] = 1
                        total_filtered_count += 1

    return noisy_labels

if __name__ == '__main__':
    run_train(args)
