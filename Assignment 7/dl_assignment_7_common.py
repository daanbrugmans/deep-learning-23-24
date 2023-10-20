# This is a file where you should put your own functions
from torch import nn
from torchvision.datasets import MNIST, FashionMNIST
from torch.utils.data import DataLoader
import torchvision
import torch


# -----------------------------------------------------------------------------
# Datasets
# -----------------------------------------------------------------------------
def get_mnis(batch_size):
    mnist = MNIST(
        root=".", download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            lambda img: img.flatten()
        ]))
    train, validation, _ = torch.utils.data.random_split(
        mnist.train_data, [5500, 500, 0])
    train = DataLoader(train, batch_size, shuffle=True)
    validation = DataLoader(validation, shuffle=True)
    test = DataLoader(mnist.test_data, batch_size, shuffle=True)
    return train, validation, test


def get_fashion_mnis():
    fashionmnist = FashionMNIST(
        root=".", download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            lambda img: img.flatten()
        ]))
    fashion_train, fashion_validation, _ = torch.utils.data.random_split(
        fashionmnist.train_data, [1000, 500, len(fashionmnist) - (1000 + 500)])


# TODO: Datasets go here.

# -----------------------------------------------------------------------------
# Network architectures
# -----------------------------------------------------------------------------

# TODO: Define network architectures here

def create_lenet(num_classes=10):
    return nn.Sequential(
        # I could not find the activation function used in the paper, but other implementations
        # of LeNet seem to use sigmoid
        nn.LazyLinear(300), nn.Sigmoid(),
        nn.Linear(300, 100), nn.Sigmoid(),
        nn.Linear(100, num_classes)
    )


def create_network(arch, **kwargs):
    # TODO: Change this function for the architectures you want to support
    if arch == 'lenet':
        return create_lenet(**kwargs)
    elif arch == 'arch2':
        return create_network_arch2(**kwargs)
    else:
        raise Exception(f"Unknown architecture: {arch}")

# -----------------------------------------------------------------------------
# Training and testing loops
# -----------------------------------------------------------------------------

# TODO: Define training, testing and model loading here

# -----------------------------------------------------------------------------
# Pruning
# -----------------------------------------------------------------------------

# TODO: Put functions related to pruning here
