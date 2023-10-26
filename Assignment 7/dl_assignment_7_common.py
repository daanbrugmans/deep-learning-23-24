# This is a file where you should put your own functions
import math
import csv
from typing import Dict, Any, Iterator
from dataclasses import dataclass

from d2l.torch import try_gpu
from torch import nn
from torch.nn import CrossEntropyLoss
from torchvision.datasets import MNIST, FashionMNIST
from torch.utils.data import DataLoader
import torchvision
import torch
import torch.nn.utils.prune
from d2l import torch as d2l


# -----------------------------------------------------------------------------
# Datasets
# -----------------------------------------------------------------------------
def get_datasets(name: str, batch_size=60, used_data=1.0) -> dict[str, DataLoader[Any]]:
    """
    :param int batch_size: batch size to use for the DataLoaders
    :param str name: name of the dataset to load. Available are 'mnist' and 'fashionmnist'
    :param float used_data: percentage of total dataset to use
    """

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

    train_data_count = math.floor(0.9 * len(train_and_val_data) * used_data)
    validation_data_count = math.ceil(0.1 * len(train_and_val_data) * used_data)
    unused_data_count = len(train_and_val_data) - train_data_count - validation_data_count

    train_data, validation_data, _ = torch.utils.data.random_split(
        train_and_val_data, [train_data_count, validation_data_count, unused_data_count]
    )

    test_count = math.ceil(len(test_data) * used_data)
    unused_data_count = len(test_data) - test_count
    test_data, _ = torch.utils.data.random_split(
        test_data, [test_count, unused_data_count]
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
def _create_lenet(in_features=784, num_classes=10):
    return nn.Sequential(
        # I could not find the activation function used in the paper, but other implementations
        # of LeNet seem to use sigmoid
        nn.Linear(in_features, 300), nn.Sigmoid(),
        nn.Linear(300, 100), nn.Sigmoid(),
        nn.Linear(100, num_classes)  # , nn.Softmax(dim=0)
    )


def _create_arch2():
    raise NotImplementedError()


def create_network(arch, **kwargs):
    # TODO: Change this function for the architectures you want to support
    if arch == 'lenet':
        net = _create_lenet(**kwargs)
    elif arch == 'arch2':
        net = _create_arch2()
    else:
        raise ValueError(f"Architecture name {arch} was not recognized")
    return net


# def get_weight_init_function(seed):
#     def init_weights_normal(m):
#         # TODO support other layer types
#         if isinstance(m, nn.Linear):
#             torch.nn.init.normal_(m.weight, mean=0.0, std=1.0)
#             torch.nn.init.zeros_(m.bias)


# -----------------------------------------------------------------------------
# Training and testing loops
# -----------------------------------------------------------------------------

# TODO: Define training, testing and model loading here
def evaluate_loss(net: nn, data_iter, loss_fn, device):
    """Evaluate the loss of a model on the given dataset.

    Copied from d2l and slightly modified"""
    metric = d2l.Accumulator(2)  # Sum of losses, no. Of examples

    for x, y in data_iter:
        x = x.to(device)
        y = y.to(device)

        y_hat = net(x)

        loss = loss_fn(y_hat, y)

        metric.add(d2l.reduce_sum(loss), d2l.size(loss))

    return metric[0] / metric[1]


def evaluate(net, evaluation_set, loss_fn, device):
    loss = evaluate_loss(net, evaluation_set, loss_fn, device)
    accuracy = d2l.evaluate_accuracy_gpu(net, evaluation_set, device=device)

    return loss, accuracy


def train(net: nn.Module, data_loaders: Dict[str, DataLoader], optimizer, loss_fn,
          device: str, model_file_name: str, learning_rate: float = 0.0012, iterations: int = 50000,
          eval_every_n_iterations=100, graph=True, keep_checkpoints=True):
    # inspired by assignment 5
    """
    Trains the model net with data from the data_loaders['train'], data_loaders['val'], data_loaders['test'].
    """
    net = net.to(device)
    optimizer = optimizer(net.parameters(), lr=learning_rate)

    if graph:
        training_progression_animator = d2l.Animator(
            xlabel='iteration',
            legend=['train loss', 'train accuracy', 'validation loss', 'validation accuracy'],
            figsize=(10, 5)
        )
    min_val_loss = float("inf")
    iteration_min_val_loss = 0
    iteration_count = 0

    with open(f'checkpoints/{model_file_name}.csv', 'x', newline='') as csvfile:
        fieldnames = ['iteration', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'test_loss', 'test_acc']
        csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csvwriter.writeheader()

        while True:
            # monitor loss, accuracy, number of samples
            # metrics = {'train': d2l.Accumulator(3), 'val': d2l.Accumulator(3)}
            for _, (x, y) in enumerate(data_loaders['train']):
                iteration_count += 1
                if iteration_count % eval_every_n_iterations == 0:
                    test_loss, test_acc = evaluate(net, data_loaders['test'], loss_fn, device)
                    val_loss, val_acc = evaluate(net, data_loaders['val'], loss_fn, device)
                    train_loss, train_acc = evaluate(net, data_loaders['train'], loss_fn, device)

                    csvwriter.writerow({'iteration': iteration_count, 'train_loss': train_loss, 'train_acc': train_acc,
                                        'val_loss': val_loss, 'val_acc': val_acc, 'test_loss': test_loss,
                                        'test_acc': test_acc})

                    if keep_checkpoints:
                        path = f"checkpoints/model-{model_file_name}-{iteration_count}.pth"
                        torch.save(net.state_dict(), path)

                    if val_loss < min_val_loss:
                        path = f"checkpoints/model-{model_file_name}-best.pth"
                        torch.save(net.state_dict(), path)
                        min_val_loss = val_loss
                        iteration_min_val_loss = iteration_count

                    if graph:
                        training_progression_animator.add(iteration_count, (train_loss, train_acc, val_loss, val_acc))

                x = x.to(device)
                y = y.to(device)

                y_hat = net(x)

                loss = loss_fn(y_hat, y)
                # accuracy = d2l.accuracy(y_hat, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # metrics['train'].add(loss * x.shape[0], accuracy, x.shape[0])

                # if iteration_count % eval_every_n_iterations == 0 and graph:
                # training_progression_animator.add(iteration_count, (
                #     metrics['train'][0] / metrics['train'][2],
                #     metrics['train'][1] / metrics['train'][2]
                # ))
                if iteration_count >= iterations:
                    break
            else:
                continue
            break

        test_loss, test_acc = evaluate(net, data_loaders['test'], loss_fn, device)
        val_loss, val_acc = evaluate(net, data_loaders['val'], loss_fn, device)
        train_loss, train_acc = evaluate(net, data_loaders['train'], loss_fn, device)

        csvwriter.writerow({'iteration': iteration_count, 'train_loss': train_loss, 'train_acc': train_acc,
                            'val_loss': val_loss, 'val_acc': val_acc, 'test_loss': test_loss,
                            'test_acc': test_acc})

    path = f"checkpoints/model-{model_file_name}-final.pth"
    torch.save(net.state_dict(), path)

    print(f'train loss {train_loss:.3f}, train accuracy {train_acc:.3f}, '
          f'val loss {val_loss:.3f}, val accuracy {val_acc:.3f}, '
          f'test loss {test_loss:.3f}, test accuracy {test_acc:.3f}')

    return train_loss, train_acc, val_loss, val_acc, test_loss, test_acc


# -----------------------------------------------------------------------------
# Pruning
# -----------------------------------------------------------------------------

def prune_random(net: nn.Sequential, p: float):
    for layer, last in is_last(net.children()):
        if last:
            if isinstance(layer, nn.Linear):
                nn.utils.prune.random_unstructured(layer, 'weight', amount=p / 2)
        if isinstance(layer, nn.Linear):
            nn.utils.prune.random_unstructured(layer, 'weight', amount=p)


# TODO: Put functions related to pruning here
def copy_prune(mask_net: nn.Sequential, apply_net: nn.Sequential):
    for mask_layer, apply_layer in zip(mask_net.children(), apply_net.children()):
        if isinstance(mask_layer, nn.Linear):
            weight_mask = mask_layer.state_dict().get('weight_mask')
            nn.utils.prune.custom_from_mask(apply_layer, 'weight', weight_mask)


def prune(net: nn.Sequential, p: float):
    """
    Prunes a network in-place, i.e., does not return a new network.

    :param net: Network to prune
    :param p: percentage of weights to pune
    """
    for layer, last in is_last(net.children()):
        if last:
            if isinstance(layer, nn.Linear):
                nn.utils.prune.l1_unstructured(layer, 'weight', amount=p / 2)
        if isinstance(layer, nn.Linear):
            nn.utils.prune.l1_unstructured(layer, 'weight', amount=p)


def is_last(i: Iterator):
    """Pass through all values from the given iterable, augmented by the
    information if there are more values to come after the current one
    (True), or if it is the last value (False).
    """
    # Copied from https://stackoverflow.com/a/1630350/21752100
    # Get an iterator and pull the first value.
    last = next(i)
    # Run the iterator to exhaustion (starting from the second value).
    for val in i:
        # Report the *previous* value (more to come).
        yield last, True
        last = val
    # Report the last value.
    yield last, False


def experiment_section_1(arch: str, dataset: str, optim, lr, fc_pruning_rate, conv_pruning_rate):
    datasets = get_datasets(dataset, used_data=0.1)
    net = create_network(arch)
    device = try_gpu()

    torch.manual_seed(1)
    name = f'experiment1-{arch}-{dataset}-{fc_pruning_rate:.3f}-{conv_pruning_rate:.3f}-random'
    prune_random(net, fc_pruning_rate)
    train(net, datasets, optim, CrossEntropyLoss(), device, name, lr, graph=False, keep_checkpoints=False)

    torch.manual_seed(2)
    net = create_network(arch)
    name = f'experiment1-{arch}-{dataset}-{fc_pruning_rate:.3f}-{conv_pruning_rate:.3f}-l1-source'
    train(net, datasets, optim, CrossEntropyLoss(), device, name, lr, iterations=5000, graph=False,
          keep_checkpoints=False)
    prune(net, fc_pruning_rate)
    net_sparse = create_network(arch)
    net_sparse = net_sparse.to(device)
    copy_prune(net, net_sparse)
    name = f'experiment1-{arch}-{dataset}-{fc_pruning_rate:.3f}-{conv_pruning_rate:.3f}-l1-target'
    train(net, datasets, optim, CrossEntropyLoss(), device, name, lr, graph=False, keep_checkpoints=False)


@dataclass
class StatisticsExperiment1:
    arch: str
    dataset: str
    prune_strategy: str
    pruned_weights: float
    early_stop_iteration: int
    acc_at_early_stop: float


def read_stats(pruned_weights, arch: str, dataset: str, prune_strategy: str):
    file = f'checkpoints/experiment1-{arch}-{dataset}-{pruned_weights:.3f}-0.000-{prune_strategy}.csv'
    with open(file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        min_val_loss = float('inf')
        early_stop_iteration = 0
        acc_at_early_stop = 0
        for row in reader:
            if float(row['val_loss']) < min_val_loss:
                early_stop_iteration = int(row['iteration'])
                min_val_loss = float(row['val_loss'])
                acc_at_early_stop = float(row['test_acc'])

    return StatisticsExperiment1(arch, dataset, prune_strategy, pruned_weights, early_stop_iteration, acc_at_early_stop)
