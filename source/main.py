import torch
from torch import nn, save, load
from torch.optim import Adam
import os

from config import get_system_device
from data_loader import get_MNIST_data
from train import train_model, load_model
from image_classifier import ImageClassifier

start_epoch = 19
total_epochs_to_train = 5 #total number of epochs that we want to train for
checkpoints_dir = os.path.join(os.getcwd(), "checkpoints")

if __name__ == '__main__':
    device = get_system_device()
    datasets = get_MNIST_data()
    clf = ImageClassifier(optimizer=Adam, loss_fn=nn.CrossEntropyLoss, lr=1e-3).to(device)
    print(clf)

    # Load model and optimizer state if resuming training
    clf, start_epoch = load_model(clf=clf, checkpoints_dir=checkpoints_dir, start_epoch=start_epoch)

    # train
    train_model(clf=clf, datasets=datasets, epochs=total_epochs_to_train, start_epoch=start_epoch, device=device,)

    # Test model
    clf.eval()
    x, y = next(iter(datasets))
    x = x.to(device)
    y = y.to(device)
    output = clf(x)
    print(output.argmax(1))
    print(y)
    print((output.argmax(1) == y).sum().item() / len(y))
