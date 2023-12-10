import torch
from torch import nn, save, load
from torch.optim import Adam
import os

from config import get_system_device
from data_loader import get_MNIST_data, get_MNIST_couples
from source.plot_data import plot_model_inference
from train import train_model, load_model
from image_classifier import ImageClassifier
from inference import test_model_performance
from source.plot_data import plot_dataset

start_epoch =11 # if set to n, loads the model from checkpoint_{n-1}.pt
total_epochs_to_train = 20  # total number of epochs that we want to train for
save_checkpoint_every_n_epochs = 10  # save a checkpoint every n epochs
checkpoints_dir = os.path.join(os.getcwd(), "checkpoints")  # where you wants your  checkpoints to be accessed and saved

import matplotlib.pyplot as plt

if __name__ == '__main__':
    device = get_system_device(print_info=True)
    train_datasets, test_datasets = get_MNIST_couples(train=True), get_MNIST_couples(train=False)

    # plot dataset
    plot_dataset(train_datasets, device=device)

    # Create model
    clf = ImageClassifier(n_digits_to_recognize=2, optimizer=Adam, loss_fn=nn.CrossEntropyLoss, lr=1e-3).to(device)

    # Load model and optimizer state if resuming training
    clf, start_epoch = load_model(clf=clf, checkpoints_dir=checkpoints_dir, start_epoch=start_epoch)

    # Train
    #train_model(clf=clf, datasets=train_datasets, epochs=total_epochs_to_train, start_epoch=start_epoch, device=device,
                #save_checkpoint_every_n_epochs=save_checkpoint_every_n_epochs, checkpoints_dir=checkpoints_dir)

    # Test model
    test_model_performance(clf, test_datasets, device)

    # Plot model inference
    plot_model_inference(clf, test_datasets, device=device)
