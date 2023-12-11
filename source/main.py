import os

from torch import nn
from torch.optim import Adam

from config import get_system_device, checkpoints_path
from data_loader import get_n_digits_dataset
from plot_data import plot_model_inference
from train import train_model, load_model
from image_classifier import ImageClassifier
from inference import test_model_performance
from plot_data import plot_dataset

#
#

# TODO: find a strategy to stop training when the model is overfitting, and give the option to the user to choose.
# TODO:add print to reassure the user while dataset is being created
#TODO: add tensorboard to requirements.txt
#TODO: add progress bars to reassure the user while training

# Hyperparameters
n_digits_in_number_to_classify = 3  # number of digits to classify
start_epoch = 0  # if set to n, loads the model from checkpoint_{n-1}.pt
total_epochs_to_train = 15  # total number of epochs that we want to train for
save_checkpoint_every_n_epochs = 1  # save a checkpoint every n epochs

if __name__ == '__main__':
    device = get_system_device(print_info=True)
    train_datasets, test_datasets = get_n_digits_dataset(n=n_digits_in_number_to_classify,
                                                         train=True), get_n_digits_dataset(
        n=n_digits_in_number_to_classify, train=False)

    # plot dataset
    plot_dataset(train_datasets)

    # Create model
    clf = ImageClassifier(n_digits_to_recognize=n_digits_in_number_to_classify, optimizer=Adam,
                          loss_fn=nn.CrossEntropyLoss, lr=1e-3).to(device)

    # Load model and optimizer state if resuming training
    clf, start_epoch = load_model(clf=clf, checkpoints_dir=checkpoints_path, start_epoch=start_epoch)

    # Train
    train_model(clf=clf, datasets=train_datasets, epochs=total_epochs_to_train, start_epoch=start_epoch, device=device,
                save_checkpoint_every_n_epochs=save_checkpoint_every_n_epochs, checkpoints_dir=checkpoints_path)

    # Test model
    test_model_performance(clf, test_datasets, device)

    # Plot model inference
    plot_model_inference(clf, test_datasets, device=device)

