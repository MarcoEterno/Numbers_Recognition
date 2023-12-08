from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


def get_MNIST_data():
    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )
    # images are 28x28 pixels, and have 10 classes (digits)
    dataset = DataLoader(training_data, batch_size=256, shuffle=True)
    return dataset
