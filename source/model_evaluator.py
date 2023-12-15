import os
from collections import defaultdict

import torch
from torch import nn
from torch.optim import Adam
from matplotlib import pyplot as plt

from config import get_system_device, checkpoints_path, custom_data_path
from data_loader import get_n_digits_dataset
from plot_data import plot_model_inference
from train import train_model, load_model
from image_classifier import ImageClassifier
from inference import test_model_performance
from plot_data import plot_dataset

max_number_of_epochs = 31
device = get_system_device(print_info=True)

def load_predifined_checkpoint(clf: ImageClassifier, predefined_checkpoints_path, epoch):
    checkpoint = torch.load(predefined_checkpoints_path)
    clf.load_state_dict(checkpoint['model_state_dict'])
    clf.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    return clf

def plot_accuracy(accuracy, n_digits_in_number_to_classify):
    plt.plot(accuracy.keys(), accuracy.values())
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy for {n_digits_in_number_to_classify} digits")
    plt.savefig(os.path.join(custom_data_path, f"accuracy_{n_digits_in_number_to_classify}_digit{'s'if n_digits_in_number_to_classify!=1 else ''}.png"))
    plt.show()

# Given a set of checkpoints, find the best one
if __name__ == '__main__':

    for n_digits_in_number_to_classify in range(2, 3):
        accuracy = defaultdict(list)
        best_model_epoch = defaultdict(int)
        print(f"Evaluating model for {n_digits_in_number_to_classify} digits")

        # Load datasets
        test_datasets = get_n_digits_dataset(n=n_digits_in_number_to_classify, train=False)

        # Create model
        clf = ImageClassifier(n_digits_to_recognize=n_digits_in_number_to_classify, optimizer=Adam,
                              loss_fn=nn.CrossEntropyLoss, lr=1e-3, print_info=False).to(device)

        for epoch in range(0, max_number_of_epochs):
            # Load predefined checkpoint if available
            if os.path.exists(os.path.join(checkpoints_path, f"{clf.numbers_to_recognize}_digit{'s' if clf.numbers_to_recognize != 1 else ''}_epoch_{epoch}.pt")):
                clf, epoch = load_model(clf=clf, checkpoints_dir=checkpoints_path, start_epoch=epoch)

                # Test model
                accuracy[epoch] = test_model_performance(clf, test_datasets)
                print(f"Accuracy for {n_digits_in_number_to_classify} digits at epoch {epoch}: {accuracy[epoch]}")

        # Plot accuracy
        plot_accuracy(accuracy, n_digits_in_number_to_classify)
        # Find best model
        best_model_epoch[n_digits_in_number_to_classify] = max(accuracy, key=accuracy.get)
        print(f"Best accuracy for {n_digits_in_number_to_classify} digits: {max(accuracy.values())}, achieved at epoch {best_model_epoch[n_digits_in_number_to_classify]}")

        # Plotting inference from the best model
        clf, epoch = load_model(clf=clf, checkpoints_dir=checkpoints_path, start_epoch=best_model_epoch[n_digits_in_number_to_classify])
        plot_model_inference(clf, get_n_digits_dataset(n=n_digits_in_number_to_classify, train=False))


"""
Keep track of best models:
- 1 digit: 99.6% accuracy, achieved at epoch 7
- 2 digits: 98.8% accuracy, achieved at epoch 9
- 3 digits: 93.7% accuracy, achieved at epoch 18
- 4 digits: 36.7% accuracy, achieved at epoch 20
in 3 and four digits the accuracy is low due to the data generation process. 
In fact, number should be generated:
1. by shuffling the digits and not concatenating them
2. by increasing the numbers of samples with increasing digits to classify.
"""
