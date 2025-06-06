import matplotlib as mpl
mpl.use('Agg')  # Avoid running into X server issues when running on servers without display
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import random
from torch import nn
from tqdm import tqdm
from IPython import display
from models.vgg import VGG_A, VGG_A_BatchNorm 
from data.loaders import get_cifar_loader

# Constants (parameters) initialization
device_id = [0,1,2,3]
batch_size = 128
num_workers = 4

# Add package dir to path 
module_path = os.path.dirname(os.getcwd())
home_path = module_path
figures_path = os.path.join(home_path, 'reports', 'figures')
models_path = os.path.join(home_path, 'reports', 'models')

# Ensure correct device
device = torch.device("cuda:{}".format(3) if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.get_device_name(3))

# # Initialize data loader and check
# train_loader = get_cifar_loader(train=True)
# val_loader = get_cifar_loader(train=False)
def get_dataloader(batch_size):
    train_loader = get_cifar_loader(train=True, batch_size=batch_size)
    val_loader = get_cifar_loader(train=False, batch_size=batch_size)
    return train_loader, val_loader

# This function is used to calculate the accuracy of the model
def get_accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# Set random seed for reproducibility
def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu': 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train(model, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=100, best_model_path=None):
    model.to(device)
    
    # Initialize lists to store iteration-wise losses and accuracies
    train_losses = []
    val_losses = []
    train_accuracies_epoch = []
    val_accuracies_epoch = []
    
    # Initialize counters and max accuracy variables
    iteration_counter = 0  # Counter for iterations
    max_val_accuracy = 0
    max_val_accuracy_epoch = 0

    batches_n = len(train_loader)

    for epoch in tqdm(range(epochs_n), unit='epoch', leave=False):
        if scheduler is not None:
            scheduler.step()

        model.train()

        running_train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # Training phase
        for data in train_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            loss.backward()
            optimizer.step()

            # Store losses and accuracies for each iteration
            train_losses.append(loss.item())

            _, predicted = torch.max(prediction, 1)
            total_train += y.size(0)
            correct_train += (predicted == y).sum().item()
            iteration_counter += 1

        # Calculate average train loss and accuracy for the epoch
        train_loss = running_train_loss / batches_n
        train_acc = (correct_train / total_train) * 100

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for data in val_loader:
                x, y = data
                x = x.to(device)
                y = y.to(device)
                prediction = model(x)
                loss = criterion(prediction, y)
                running_val_loss += loss.item()

                _, predicted = torch.max(prediction, 1)
                total_val += y.size(0)
                correct_val += (predicted == y).sum().item()

        # Calculate average validation loss and accuracy
        val_loss = running_val_loss / len(val_loader)
        val_acc = (correct_val / total_val) * 100

        # Store epoch-wise accuracies
        train_accuracies_epoch.append(train_acc)
        val_accuracies_epoch.append(val_acc)
        val_losses.append(val_loss)

        # # Print stats for current epoch
        # tqdm.write(f"Epoch {epoch+1}/{epochs_n}: Train Loss: {train_loss:.4f}| Train Accuracy: {train_acc:.2f}%| Val Loss: {val_loss:.4f}| Val Accuracy: {val_acc:.2f}%")

        # Save best model based on validation accuracy
        if val_acc > max_val_accuracy:
            max_val_accuracy = val_acc
            max_val_accuracy_epoch = epoch
            if best_model_path:
                os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
                torch.save(model.state_dict(), best_model_path)

    return train_losses, val_losses, train_accuracies_epoch, val_accuracies_epoch


