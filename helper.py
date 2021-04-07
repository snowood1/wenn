import scipy.stats
import scipy.io
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms
from sklearn.metrics import auc, roc_curve


if torch.cuda.is_available():
    device = torch.device('cuda')
    CUDA = True
else:
    device = torch.device('cpu')
    CUDA = False

# You can modify these parameters for dataloader according to your environments.
kwargs = {'num_workers': 8, 'pin_memory': True} if CUDA else {}
# This is the root path of all the datasets. Mkdir a dataset named 'datasets' under your project folder.
Root = 'datasets/'


normal_transform = transforms.ToTensor()

# Black and white images apply simple augmentation.
augment_transform_bw = transforms.Compose([
    transforms.RandomCrop(28, padding=2),
    transforms.ToTensor()])
augment_transform_bw = transforms.RandomChoice([augment_transform_bw, normal_transform])

# Colorful images apply random flip and crop augmentation.
augment_transform_color = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor()])
augment_transform_color = transforms.RandomChoice([augment_transform_color, normal_transform])

# SVHN doesn't apply random horizontal flip
augment_transform_svhn = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor()])
augment_transform_svhn = transforms.RandomChoice([augment_transform_svhn, normal_transform])


# Generate dataloaders of different datasets.
def get_data(dataname, batch_size, use_augment=True, use_split=True, split=10000):

    if dataname in ['MNIST', 'FashionMNIST']:
        train_transform = augment_transform_bw if use_augment else normal_transform

    if dataname in ['CIFAR5', 'CIFAR10', 'CIFAR100', 'LSUN']:
        train_transform = augment_transform_color if use_augment else normal_transform

    if dataname in ['SVHN']:
        train_transform = augment_transform_svhn if use_augment else normal_transform


    if dataname == 'SVHN':
        dataset_kwargs = [{'split': 'train'}, {'split': 'train'}, {'split': 'test'}]
    elif dataname == 'LSUN':
        dataset_kwargs = [{'classes': 'train'}, {'classes': 'val'}, {'classes': 'test'}]
    elif dataname in ['CIFAR5', 'CIFAR10', 'CIFAR100', 'FashionMNIST', 'MNIST']:
        dataset_kwargs = [{'train': True}, {'train': True}, {'train': False}]

    train_data = getattr(datasets, dataname)(Root + str(dataname), transform=train_transform, download=True, **dataset_kwargs[0])
    valid_data = getattr(datasets, dataname)(Root + str(dataname), transform=normal_transform, download=True, **dataset_kwargs[1])
    test_data = getattr(datasets, dataname)(Root + str(dataname), transform=normal_transform, download=True, **dataset_kwargs[2])

    if use_split:
        train_num = len(train_data) - split
        train_data, valid_data = torch.utils.data.random_split(train_data, [train_num, split])
    #
    # kwargs = {'num_workers': 16, 'pin_memory': True} if CUDA else {}  #########TODO

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,  shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size,  shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, **kwargs)

    return train_loader, valid_loader, test_loader


# This function converts the first 0 - 4 class in CIFAR-10 as in-distribution data.
#  and the last five classes 5-9 as out-of-distribution data.

def get_cifar5(batch_size, split, use_augment, validation):
    dataname = 'CIFAR10'
    train_transform = augment_transform_color if use_augment else normal_transform

    train_data = getattr(datasets, dataname) \
        (Root + str(dataname), train=True, download=True, transform=train_transform)

    test_data = getattr(datasets, dataname)(Root + str(dataname), train=False, download=True,
                                            transform=normal_transform)

    train_img, train_targets = np.array(train_data.data), np.array(train_data.targets)
    # valid_img, valid_targets = np.array(valid_data.data), np.array(valid_data.targets)
    test_img, test_targets = np.array(test_data.data), np.array(test_data.targets)

    # 0~ 4 in class train and valid
    train_in_idx = np.where(train_targets < 5)[0]  # training in-class index

    if validation:
        indices = train_in_idx
        np.random.shuffle(indices)
        train_idx, valid_idx = indices[split:], indices[:split]
    else:
        # split = 0
        train_idx = train_in_idx

    # 0~4 in class test data
    test_in_idx = np.where(test_targets < 5)[0]
    # 5~9 ood
    test_ood_idx = np.where(test_targets >= 5)[0]

    # kwargs = {'num_workers': 8, 'pin_memory': True} if CUDA else {}  #####

    train_loader = torch.utils.data.DataLoader(Subset(train_data, train_idx), batch_size=batch_size,
                                               shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(Subset(train_data, valid_idx), batch_size=batch_size,
                                               shuffle=False, **kwargs) if validation else None
    test_loader = torch.utils.data.DataLoader(Subset(test_data, test_in_idx), batch_size=batch_size,
                                              shuffle=False, **kwargs)
    ood_loader = torch.utils.data.DataLoader(Subset(test_data, test_ood_idx), batch_size=batch_size,
                                             shuffle=False, **kwargs)
    return train_loader, valid_loader, test_loader, ood_loader


# Generate not-mnist dataloader. An OOD dataset for evaluating FashionMNIST.
# Please download the dataset in http://yaroslavvb.com/upload/notMNIST/

def get_notmnist(batch_size, split):
    dataset = scipy.io.loadmat(Root + "notMNIST_small.mat")
    data = np.array(dataset["images"]).transpose() / 255
    num = len(data)
    indices = list(range(num))
    np.random.shuffle(indices)
    data = data[indices[:split]]
    data = torch.tensor(data)
    data = data.unsqueeze(1).float()
    notmnist_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, **kwargs)
    return notmnist_loader

# For LSUN, you can use LSUN_resize or LSUN_classroom
# This function is used to read and generate LSUN_resize.
# Please download LSUN_resize in https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz

def get_lsun(batch_size, split):
    data = datasets.ImageFolder(root= Root + 'LSUN_resize',
                         transform=normal_transform)
    num = len(data)
    indices = list(range(num))
    np.random.shuffle(indices)
    split_idx = indices[:split]
    lsun_loader = torch.utils.data.DataLoader(Subset(data, split_idx), batch_size=batch_size, **kwargs)
    return lsun_loader


# Calculate dissonance of a vector of alpha #
def getDisn(alpha):
    evi = alpha - 1
    s = torch.sum(alpha, axis=1, keepdims=True)
    blf = evi / s
    idx = np.arange(alpha.shape[1])
    diss = 0
    Bal = lambda bi, bj: 1 - torch.abs(bi - bj) / (bi + bj + 1e-8)
    for i in idx:
        score_j_bal = [blf[:, j] * Bal(blf[:, j], blf[:, i]) for j in idx[idx != i]]
        score_j = [blf[:, j] for j in idx[idx != i]]
        diss += blf[:, i] * sum(score_j_bal) / (sum(score_j) + 1e-8)
    return diss


# Calculate entropy of a vector of probability
def cal_entropy(p):
    if type(p) == torch.Tensor:
        return (-p * torch.log(p + 1e-8)).sum(1)
    else:
        return (-p * np.log(p + 1e-8)).sum(1)


# Evaluation:  get roc curve from a set of normal scores and anormal scores.
def get_pr_roc(normal_score, anormal_score):

    if type(normal_score) == pd.core.series.Series:
        normal_score = normal_score.iloc[0]
    if type(anormal_score) == pd.core.series.Series:
        anormal_score = anormal_score.iloc[0]

    truth = np.zeros((len(normal_score) + len(anormal_score)))
    truth[len(normal_score):] = 1

    score = np.concatenate([normal_score, anormal_score])

    fpr, tpr, _ = roc_curve(truth, score, drop_intermediate=False)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


# def fgsm_attack(image, epsilon, data_grad):
#     # Collect the element-wise sign of the data gradient
#     sign_data_grad = data_grad.sign()
#     # Create the perturbed image by adjusting each pixel of the input image
#     perturbed_image = image + epsilon*sign_data_grad
#     # Adding clipping to maintain [0,1] range
#     perturbed_image = torch.clamp(perturbed_image, 0, 1)
#     # Return the perturbed image
#     return perturbed_image
