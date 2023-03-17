from typing import List, Tuple
from models.AdaptableResNetCifar import AdaptableResNetCifar, createModel
from models.AdaptableViT import AdaptableVisionTransformer
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from cutout import Cutout
import os

def get_model(depth: int, num_classes: int, device: torch.device, dataset_name = "") -> AdaptableResNetCifar:
    net = createModel(depth, dataset_name, num_classes)
    net = net.to(device)
    return net

def get_CIFAR10_data():

    # bypass ssl problem by https -> http
    torchvision.datasets.CIFAR10.url="http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

    standard_trans =  [
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.4914, 0.4824, 0.4467],
                            std=[0.2471, 0.2435, 0.2616])
    ]

    aug_trans = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]

    cutout_trans = [Cutout(n_holes=1, length= 16)]

    download = False if os.path.exists('./data') else True

    transform = transforms.Compose(standard_trans + aug_trans + cutout_trans)

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=download, transform=transform)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=download, transform = transforms.Compose(standard_trans))
    
    return trainset, testset

def get_CIFAR10(batch_size: int, val_size: int, train_size: int, federated = False) -> Tuple[torch.utils.data.DataLoader,
                                                                          torch.utils.data.DataLoader,
                                                                          torch.utils.data.DataLoader]:

    assert val_size + train_size == 50000

    dset, testset = get_CIFAR10_data()

    trainset, valset = torch.utils.data.random_split(dset, [train_size, val_size])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle= not federated , num_workers=1)

    val_batch = val_size if not federated else batch_size

    valloader = torch.utils.data.DataLoader(valset, batch_size = val_batch, 
                                            shuffle = not federated, num_workers = 1)

    test_batch = len(testset) if not federated else batch_size

    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch,
                                            shuffle=False, num_workers=1)

    return trainloader, valloader, testloader

def get_CIFAR100_data():

    # bypass ssl problem by https -> http
    torchvision.datasets.CIFAR100.url="http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"

    standard_trans =  [
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.4914, 0.4824, 0.4467],
                            std=[0.2471, 0.2435, 0.2616])
    ]

    aug_trans = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]

    cutout_trans = [Cutout(n_holes=1, length= 16)]

    data_path = '/mnt/disk1/fadhel/img_classification_pk_pytorch/data'
    download = True

    transform = transforms.Compose(standard_trans + aug_trans + cutout_trans)

    trainset = torchvision.datasets.CIFAR100(root=data_path, train=True,
                                            download=download, transform=transform)

    testset = torchvision.datasets.CIFAR100(root=data_path, train=False,
                                        download=download, transform = transforms.Compose(standard_trans))
    
    return trainset, testset

def get_CIFAR100(batch_size: int, val_size: int, train_size: int, federated = False) -> Tuple[torch.utils.data.DataLoader,
                                                                          torch.utils.data.DataLoader,
                                                                          torch.utils.data.DataLoader]:

    assert val_size + train_size == 50000

    dset, testset = get_CIFAR100_data()

    trainset, valset = torch.utils.data.random_split(dset, [train_size, val_size])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle= not federated , num_workers=1)

    val_batch = val_size if not federated else batch_size

    valloader = torch.utils.data.DataLoader(valset, batch_size = val_batch, 
                                            shuffle = not federated, num_workers = 1)

    test_batch = len(testset) if not federated else batch_size

    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch,
                                            shuffle=False, num_workers=1)

    return trainloader, valloader, testloader

def get_model_fractions(net: AdaptableResNetCifar) -> List[float]:

    active_blocks = net.active_blocks

    net.reconfigure_blocks(net.max_blocks)
    total_blocks = net.active_blocks

    model_fractions = []

    for n_blocks in range(total_blocks + 1):
        net.reconfigure_blocks(n_blocks)
        model_fractions.append(net.model_fraction)
    
    net.reconfigure_blocks(active_blocks)
    return model_fractions

def load_model(path: str, depth: int, num_classes: int, device: torch.device, dataset_name = '') -> AdaptableResNetCifar:
    net = createModel(depth, dataset_name, num_classes)
    net.load_state_dict(torch.load(path))
    net = net.to(device)
    return net

def load_transformer(path: str, device: torch.device, **kwargs):
    net = AdaptableVisionTransformer(**kwargs)
    net.load_state_dict(torch.load(path))
    net = net.to(device)
    return net

def get_model_size(net: nn.Module):
    # copied from https://discuss.pytorch.org/t/finding-model-size/130275
    param_size = 0
    for param in net.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in net.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb