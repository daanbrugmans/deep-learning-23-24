# This is a file where you should put your own functions
from typing import Dict, Any

from torch import nn
from torchvision.datasets import MNIST, FashionMNIST
from torch.utils.data import DataLoader
import torchvision
import torch
from d2l import torch as d2l


# -----------------------------------------------------------------------------
# Datasets
# -----------------------------------------------------------------------------
def get_dataset(name: str, batch_size=60) -> tuple[DataLoader, DataLoader, DataLoader]:
    image_transformations = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda image: torch.flatten(image))
    ])
    
    if name == "mnist":
        train_and_val_data = MNIST(root=".", train=True, download=True, transform=image_transformations)
        test_data = MNIST(root=".", train=False, download=True, transform=image_transformations)
    elif name == "fashionmnist":
        train_and_val_data = FashionMNIST(root=".", train=True, download=True, transform=image_transformations)
        test_data = FashionMNIST(root=".", train=False, download=True, transform=image_transformations)
    else:
        raise ValueError(f"Dataset name {name} was not recognized")
            
    train_data_count = 450
    validation_data_count = 50
    unused_data_count = len(train_and_val_data) - train_data_count - validation_data_count

    train_data, validation_data, _ = torch.utils.data.random_split(
        train_and_val_data, [train_data_count, validation_data_count, unused_data_count]
    )

    return {
        "train": DataLoader(train_data, batch_size, shuffle=True),
        "val": DataLoader(validation_data, batch_size),
        "test": DataLoader(test_data, batch_size)
    }

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
def evaluate_loss(net, data_iter, loss, device):
    """Evaluate the loss of a model on the given dataset.

    Copied from d2l and slightly modified"""

    metric = d2l.Accumulator(2)  # Sum of losses, no. Of examples
    for X, y in data_iter:
        X = X.to(device)
        y = y.to(device)
        
        out = net(X)
        
        l = loss(out, y)
        
        metric.add(d2l.reduce_sum(l), d2l.size(l))
        
    return metric[0] / metric[1]


def evaluate(net, evaluation_set, loss, device):
    loss = evaluate_loss(net, evaluation_set, loss, device)
    acc = d2l.evaluate_accuracy_gpu(net, evaluation_set)
    return loss, acc


def train(net: nn.Module, data_loaders, optimizer, loss_fn, device, name: str, learning_rate: float = 0.0012,
          epochs: int = 50000):
    # inspired by assignment 5
    """
    Trains the model net with data from the data_loaders['train'], data_loaders['val'], data_loaders['test'].
    """
    net = net.to(device)
    optimizer = optimizer(net.parameters(), lr=learning_rate)
    animator = d2l.Animator(xlabel='epoch',
                            legend=['train loss', 'train acc', 'validation loss', 'validation acc'],
                            figsize=(10, 5))
    timer = {'train': d2l.Timer(), 'val': d2l.Timer()}
    eval_every_n_epochs = 2

    for epoch in range(epochs):
        # monitor loss, accuracy, number of samples
        metrics = {'train': d2l.Accumulator(2), 'val': d2l.Accumulator(2)}

        for phase in ('train', 'val'):
            if phase == 'val' and not epoch % eval_every_n_epochs == 0:
                continue
            # switch network to train/eval mode
            net.train(phase == 'train')

            for i, (x, y) in enumerate(data_loaders[phase]):
                timer[phase].start()

                # move to device
                x = x.to(device)
                y = y.to(device)

                y_hat = net(x)

                loss = loss_fn(y_hat, y)

                if phase == 'train':
                    # compute gradients and update weights
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if epoch % eval_every_n_epochs == 0:
                    loss, acc = evaluate(net, data_loaders[phase], loss_fn, device)
                    metrics[phase].add(loss, acc)

                timer[phase].stop()

        if epoch % eval_every_n_epochs == 0:
            path = f"checkpoints/model-{name}-{epoch}.pth"
            torch.save(net.state_dict(), path)

            animator.add(epoch + 1,
                         (metrics['train'][0],
                          metrics['train'][1],
                          metrics['val'][0],
                          metrics['val'][1]))

    loss, acc = evaluate(net, data_loaders['test'], loss_fn, device)

    train_loss = metrics['train'][0]
    train_acc = metrics['train'][1]
    val_loss = metrics['val'][0]
    val_acc = metrics['val'][1]

    print(f'train loss {train_loss:.3f}, train acc {train_acc:.3f}, '
          f'val loss {val_loss:.3f}, val acc {val_acc:.3f}, '
          f'test loss {loss:.3f}, test acc {acc:.3f}')

# -----------------------------------------------------------------------------
# Pruning
# -----------------------------------------------------------------------------

# TODO: Put functions related to pruning here
