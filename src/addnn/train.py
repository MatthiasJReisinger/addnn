import addnn
import addnn.model
import torch
import torchvision
import time
import copy
import os
from tqdm import tqdm
from typing import cast, List, Any
from torchvision import transforms


def train(model: addnn.model.Model, train_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader,
        epochs: int) -> None:
    """ Train an ADDNN model on the given training data set and evaluate it on the given test dataset.

    Args:
        model (addnn.Model): The model to train.
        train_loader (torch.utils.data.DataLoader): The data loader for the training dataset.
        test_loader (torch.utils.data.DataLoader): The data loader for the test dataset.
        epochs (int): The number of epochs to use for training.
    """

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)

    number_of_exits = sum([layer.exit_branch is not None for layer in model.layers])

    exits = []
    for layer in model.layers:
        if layer.exit_branch is not None:
            exits.append(layer.exit_branch)

    for exit_index in range(number_of_exits):
        print("Train exit {}".format(exit_index))

        # only optimize parameters of the exit classifier
        exit_parameters = filter(lambda parameter: parameter.requires_grad, exits[exit_index].parameters())

        optimizer = torch.optim.SGD(exit_parameters, lr=0.001, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(1, epochs + 1):
            print("Exit {}, epoch {}".format(exit_index, epoch))
            apply_training_set(model, train_loader, criterion, optimizer, device, exit_index)
            apply_test_set(model, test_loader, criterion, device, exit_index)
            scheduler.step()


def apply_training_set(model: addnn.model.Model, train_loader: torch.utils.data.DataLoader, criterion: torch.nn.Module, optimizer: torch.optim.Optimizer,
        device: torch.device, exit_index: int) -> None:
    model.train()
    model.to(device)

    total_loss = 0.0
    total_corrects = torch.tensor(0)

    for inputs, labels in tqdm(train_loader, leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = model(inputs)[exit_index]
            _, predictions = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        total_corrects += torch.sum(predictions == labels.data)

    epoch_loss = total_loss / len(cast(List[Any], train_loader.dataset))
    epoch_accuracy = total_corrects.double() / len(cast(List[Any], train_loader.dataset))

    print('Training Loss: {:.4f}, Accuracy: {:.4f}'.format(epoch_loss, epoch_accuracy))


def apply_test_set(model: addnn.model.Model, test_loader: torch.utils.data.DataLoader, criterion: torch.nn.Module, device: torch.device, exit_index: int) -> None:
    model.eval()
    model.to(device)

    total_loss = 0.0
    total_corrects = torch.tensor(0)

    for inputs, labels in tqdm(test_loader, leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)[exit_index]
            _, predictions = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        total_loss += loss.item() * inputs.size(0)
        total_corrects += torch.sum(predictions == labels.data)

    epoch_loss = total_loss / len(cast(List[Any], test_loader.dataset))
    epoch_accuracy = total_corrects.double() / len(cast(List[Any], test_loader.dataset))

    print('Test Loss: {:.4f}, Accuracy: {:.4f}'.format(epoch_loss, epoch_accuracy))
