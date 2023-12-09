import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import Tensor

from config import get_system_device


def plot_model_inference(model, test_datasets, n_rows=5, n_cols=10, device=get_system_device()):
    # Get a batch of images and labels from the dataset
    images, labels = next(iter(test_datasets))
    print(images, labels)
    images = Tensor.cpu(images)
    # Convert the tensor to numpy for visualization
    images_numpy = images.detach().to_numpy()

    # Move the input tensors to the device
    tensor_images = images.to(get_system_device())
    # Plot the images
    fig = plt.figure(figsize=(25, 4))

    n_rows = 5
    n_cols = 10
    # Display 20 images
    for idx in np.arange(n_rows * n_cols):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(images_numpy[idx]), cmap='gray')
        # Get the model's prediction
        prediction = model(tensor_images[idx].unsqueeze(0))
        # Print out the predicted label for each image
        ax.set_title(str(prediction.argmax(1).item()),
                     color=("limegreen" if prediction.argmax(1).item() == labels[idx] else "red"), fontsize=25)

    plt.show()


def plot_dataset(datasets, device=get_system_device()):
    # Get a batch of images and labels from the dataset
    images, labels = next(iter(datasets))
    print(images, labels)

    # Convert the tensor to numpy for visualization
    images_numpy = images.cpu().numpy()

    # Plot the images
    fig = plt.figure(figsize=(6, 4))

    n_rows = 5
    n_cols = 5
    # Display n_rows*n_cols images
    for idx in np.arange(n_rows * n_cols):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(images_numpy[idx]), cmap='gray')
        # Print out the predicted label for each image
        ax.set_title(labels[idx].cpu().numpy(),  fontsize=25)

    plt.subplots_adjust(hspace=1)
    plt.show()
