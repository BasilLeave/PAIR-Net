import os
import numpy as np
from PIL import Image
import torch
import copy
from torch.utils.data import Dataset
from torchvision import transforms
import random
from pathlib import Path
import cv2

import matplotlib.pyplot as plt


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET,
                      image_weight: float = 0.5) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :param image_weight: The final result is image_weight * img + (1-image_weight) * mask.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    if image_weight < 0 or image_weight > 1:
        raise Exception(
            f"image_weight should be in the range [0, 1].\
                Got: {image_weight}")

    cam = (1 - image_weight) * heatmap + image_weight * img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

AID_CATEGORY = ['airplane', 'bare soil', 'buildings', 'cars', 'chaparral', 'court', 'dock', 'field', 'grass', 'mobile home', 'pavement', 'sand', 'sea', 'ship', 'tanks', 'trees', 'water']
UCMERCED_CATEGORY = ['airplane', 'bare soil', 'buildings', 'cars', 'chaparral', 'court', 'dock', 'field', 'grass', 'mobile home', 'pavement', 'sand', 'sea', 'ship', 'tanks', 'trees', 'water']
def get_transforms(image_size):
    '''
    Returns image transforms.
    '''
    # print(image_size)
    
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    tx = {}
    
    tx['train'] = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    tx['test'] = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    tx['val'] = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    
    return tx


class ds_multilabel(Dataset):
    def __init__(self, phase, args, tx):
        # self.dataset_name = args.dataset
        self.num_class = args.num_class
        self.path_to_dataset = args.path_to_dataset

        self.phase = phase
        self.cam_to_dataset = getattr(args, 'cam_to_dataset', None)
        self.get_cam = getattr(args, 'get_cam', False)

        if phase == 'train' or phase == 'test':
            self.image_paths = np.load(os.path.join(self.path_to_dataset, f'formatted_{phase}_images.npy'))#image_ids
            self.label_matrix = np.load(os.path.join(self.path_to_dataset, f'formatted_{phase}_labels.npy'))#label_matrix

        elif phase == 'val':
            '''from train dataset select 100 images for validation'''
            image_paths = np.load(os.path.join(self.path_to_dataset, f'formatted_train_images.npy'))
            label_matrix = np.load(os.path.join(self.path_to_dataset, f'formatted_train_labels.npy'))

            # select the indices in the args.matrix that are not all zeros
            indices = np.where(np.sum(args.matrix, axis=1) > 0)[0]
            # select random the 5 indices
            num_to_select = min(15, len(indices))
            random_indices = np.random.choice(indices, num_to_select, replace=False)

            self.image_paths = image_paths[random_indices]
            self.label_matrix = label_matrix[random_indices]
            self.indices = random_indices

        self.tx = tx[phase]

    def inject_mix_noise(self, args):
        self.false_positive_label_matrix = np.zeros_like(self.label_matrix)
        self.true_positive_label_matrix = copy.deepcopy(self.label_matrix)
        self.true_negative_label_matrix = copy.deepcopy(1-self.label_matrix)
        self.false_negative_label_matrix = np.zeros_like(self.label_matrix)

        print('current positive labels:', np.sum(self.label_matrix))
        print('target noise number:', int(np.sum(self.label_matrix) * args.noise_rate / 100.0))

        clip_logits = np.load(os.path.join(args.path_to_dataset, 'clip_logits.npy'))
        pos_logits = clip_logits[self.label_matrix == 1]
        neg_logits = clip_logits[self.label_matrix == 0]
        pos_labels_num = int(np.sum(self.label_matrix))
        num_additive = int(pos_labels_num * args.noise_rate / 100.0)
        num_subtractive = int(pos_labels_num * args.noise_rate / 100.0)
        # sort neg_logits from large to small and select the num_additive-th element as the threshold
        neg_logits = np.sort(neg_logits)[::-1]
        additive_threshold = neg_logits[num_additive]
        # sort pos_logits from small to large and select the num_subtractive-th element as the threshold
        pos_logits = np.sort(pos_logits)
        subtractive_threshold = pos_logits[num_subtractive]

        potential_additive_indices = []
        potential_subtractive_indices = []
        actual_additive = 0
        actual_subtractive = 0

        for i in range(self.label_matrix.shape[0]):
            for j in range(self.label_matrix.shape[1]):
                if self.label_matrix[i][j] == 0 and clip_logits[i][j] > additive_threshold:
                    self.label_matrix[i][j] = 1
                    actual_additive += 1
                    self.false_positive_label_matrix[i][j] = 1

                elif self.label_matrix[i][j] == 0 and clip_logits[i][j] == additive_threshold:
                    potential_additive_indices.append((i, j))
                elif self.label_matrix[i][j] == 1 and clip_logits[i][j] < subtractive_threshold:
                    self.label_matrix[i][j] = 0
                    actual_subtractive += 1
                    self.false_negative_label_matrix[i][j] = 1

                elif self.label_matrix[i][j] == 1 and clip_logits[i][j] == subtractive_threshold:
                    potential_subtractive_indices.append((i, j))

        # randomly select the remaining indices to add noise
        random.shuffle(potential_additive_indices)
        random.shuffle(potential_subtractive_indices)
        if actual_additive < num_additive:
            for i, j in potential_additive_indices:
                self.label_matrix[i][j] = 1
                actual_additive += 1
                self.false_positive_label_matrix[i][j] = 1
                if actual_additive == num_additive:
                    break
        if actual_subtractive < num_subtractive:
            for i, j in potential_subtractive_indices:
                self.label_matrix[i][j] = 0
                actual_subtractive += 1
                self.false_negative_label_matrix[i][j] = 1
        
                if actual_subtractive == num_subtractive:
                    break

        assert actual_additive == num_additive == np.sum(self.false_positive_label_matrix)
        assert actual_subtractive == num_subtractive == np.sum(self.false_negative_label_matrix)
        print(f'Noise rate : {args.noise_rate}')
        print(f'Number of additive noise : {actual_additive}')
        print(f'Number of subtractive noise : {actual_subtractive}')

        print('current positive labels:', np.sum(self.label_matrix))
        print('current negative labels:', np.sum(1 - self.label_matrix))

        with open(args.logger_file, 'w') as logger:
            logger.write(f'Noise rate : {args.noise_rate}\n')
            logger.write(f'Number of additive noise : {actual_additive}\n')
            logger.write(f'Number of subtractive noise : {actual_subtractive}\n')

    def inject_additive_noise(self, args):
        # store the aditive noise label matrix
        self.false_positive_label_matrix = np.zeros_like(self.label_matrix)
        self.true_positive_label_matrix = copy.deepcopy(self.label_matrix)

        clip_logits = np.load(os.path.join(args.path_to_dataset, 'clip_logits.npy'))
        neg_logits = clip_logits[self.label_matrix == 0]
        pos_labels_num = int(np.sum(self.label_matrix))
        num_additive = int(pos_labels_num * args.noise_rate / 100.0)
        # sort neg_logits from large to small and select the num_additive-th element as the threshold
        neg_logits = np.sort(neg_logits)[::-1]
        additive_threshold = neg_logits[num_additive]

        potential_additive_indices = []
        actual_additive = 0

        for i in range(self.label_matrix.shape[0]):
            for j in range(self.label_matrix.shape[1]):
                if self.label_matrix[i][j] == 0 and clip_logits[i][j] > additive_threshold:
                    self.label_matrix[i][j] = 1
                    actual_additive += 1
                    self.false_positive_label_matrix[i][j] = 1

                elif self.label_matrix[i][j] == 0 and clip_logits[i][j] == additive_threshold:
                    potential_additive_indices.append((i, j))

        # randomly select the remaining indices to add noise
        random.shuffle(potential_additive_indices)
        if actual_additive < num_additive:
            for i, j in potential_additive_indices:
                self.label_matrix[i][j] = 1
                actual_additive += 1
                self.false_positive_label_matrix[i][j] = 1
                if actual_additive == num_additive:
                    break

        assert actual_additive == num_additive == np.sum(self.false_positive_label_matrix)
        print(f'Noise rate : {args.noise_rate}')
        print(f'Number of additive noise : {actual_additive}')
        print(f'Number of true positive : {np.sum(self.true_positive_label_matrix)}')
        print(f'Number of positive lable: {np.sum(self.label_matrix)}')
        with open(args.logger_file, 'w') as logger:
            logger.write(f'Noise rate : {args.noise_rate}\n')
            logger.write(f'Number of additive noise : {actual_additive}\n')

    def inject_subtractive_noise(self, args):
        self.true_negative_label_matrix = copy.deepcopy(1-self.label_matrix)
        self.false_negative_label_matrix = np.zeros_like(self.label_matrix)

        clip_logits = np.load(os.path.join(args.path_to_dataset, 'clip_logits.npy'))
        pos_logits = clip_logits[self.label_matrix == 1]
        pos_labels_num = int(np.sum(self.label_matrix))
        num_subtractive = int(pos_labels_num * args.noise_rate / 100.0)
        # sort pos_logits from small to large and select the num_subtractive-th element as the threshold
        pos_logits = np.sort(pos_logits)
        subtractive_threshold = pos_logits[num_subtractive]

        potential_subtractive_indices = []
        actual_subtractive = 0

        for i in range(self.label_matrix.shape[0]):
            for j in range(self.label_matrix.shape[1]):
                if self.label_matrix[i][j] == 1 and clip_logits[i][j] < subtractive_threshold:
                    self.label_matrix[i][j] = 0
                    actual_subtractive += 1
                    self.false_negative_label_matrix[i][j] = 1

                elif self.label_matrix[i][j] == 1 and clip_logits[i][j] == subtractive_threshold:
                    potential_subtractive_indices.append((i, j))

        # randomly select the remaining indices to add noise
        random.shuffle(potential_subtractive_indices)
        if actual_subtractive < num_subtractive:
            for i, j in potential_subtractive_indices:
                self.label_matrix[i][j] = 0
                actual_subtractive += 1
                self.false_negative_label_matrix[i][j] = 1

                if actual_subtractive == num_subtractive:
                    break

        assert actual_subtractive == num_subtractive == np.sum(self.false_negative_label_matrix)
        print(f'Noise rate : {args.noise_rate}')
        print(f'Number of subtractive noise : {actual_subtractive}')
        print(f'Number of true negative : {np.sum(self.true_negative_label_matrix)}')
        print(f'Number of negative lable: {np.sum(1-self.label_matrix)}')
        with open(args.logger_file, 'w') as logger:
            logger.write(f'Noise rate : {args.noise_rate}\n')
            logger.write(f'Number of subtractive noise : {actual_subtractive}\n')

    def inject_noise(self, args):

        clip_logits = np.load(os.path.join(args.path_to_dataset, 'clip_logits.npy'))
        pos_logits = clip_logits[self.label_matrix == 1]

        subtractive_threshold = np.percentile(pos_logits, args.noise_rate)
        additive_threshold = np.percentile(pos_logits, 100 - args.noise_rate)

        num_subtractive = 0
        num_additive = 0

        for i in range(self.label_matrix.shape[0]):
            for j in range(self.label_matrix.shape[1]):
                if self.label_matrix[i][j] == 0 and clip_logits[i][j] >= additive_threshold:
                    self.label_matrix[i][j] = 1  # additive noise
                    num_additive += 1
                elif self.label_matrix[i][j] == 1 and clip_logits[i][j] <= subtractive_threshold:
                    self.label_matrix[i][j] = 0  # subtractive noise
                    num_subtractive += 1


        print(f'Noise rate : {args.noise_rate}')
        print(f'Number of additive noise : {num_additive}')
        print(f'Number of subtractive noise : {num_subtractive}')


    def remove_noisy_labels(self, noisy_labels):
        '''remove the false positive labels from the label matrix'''
        '''noisy_labels: np.array of false positive labels'''
        self.label_matrix[noisy_labels == 1] = 0


    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        # print('image_path:', image_path)
        p = Path(image_path)
        file_stem = p.stem
        parent_dir_name = p.parent.name
        with Image.open(image_path) as I_raw:
            I = self.tx(I_raw)
        label = torch.FloatTensor(np.copy(self.label_matrix[idx, :]))
        if self.phase == 'val':
            idx = self.indices[idx]

        if self.get_cam:
            if self.phase == 'test':
                return I, label, idx, np.empty((0,), dtype=np.float32)

            file_name = os.path.join(self.cam_to_dataset, parent_dir_name, file_stem + '.npy')
            if not os.path.exists(file_name):
                raise FileNotFoundError(
                    f'CAM file not found for {self.phase} sample: {file_name}. '
                    f'Image path: {image_path}'
                )
            cam = np.load(file_name)
            return I, label, idx, cam
        else:
            return I, label, idx

def get_dataset(args):
    tx = get_transforms(args.img_size)
    dataset_train = ds_multilabel('train', args, tx)
    dataset_test = ds_multilabel('test', args, tx)
    return {'train': dataset_train, 'test': dataset_test}

def get_val_dataset(args):
    tx = get_transforms(args.img_size)
    dataset_val = ds_multilabel('val', args, tx)
    return dataset_val


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


if __name__ == '__main__':
    class Config:
        pass

    args = Config()
    args.cam_to_dataset = None
    args.noise_rate = 10
    args.img_size = 448
    args.num_class = 17
    args.dataset = 'UCMerced'
    args.seed = 42
    args.save_path = None
    args.bsize = 16
    args.num_workers = 4
    args.lr = 1e-4
    args.scheme = 'BCE'
    args.num_epochs = 50
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.logger_file = './logger.txt'

    if args.dataset == 'AID':
        category = AID_CATEGORY
        num_class = 17
        num_train = 2400
        args.bsize = 16
        args.path_to_dataset = '/data/user9/datasets/MLC/AID_multilabel'
    elif args.dataset == 'MLRSNet':
        category = MLRSNET_CATEGORY
        num_class = 60
        num_train = 87319
        args.path_to_dataset = '/data/user9/datasets/MLC/MLRSNet'
        args.bsize = 64
    elif args.dataset == 'UCMerced':
        category = UCMERCED_CATEGORY
        num_class = 17
        num_train = 1680
        args.bsize = 16
        args.path_to_dataset = '/data/user9/datasets/MLC/UCMerced_LandUse'


    dataset = get_dataset(args)

    # IDMN
    dataset['train'].inject_noise(args)
    # DC-IDMN 
    dataset['train'].inject_mix_noise(args)



