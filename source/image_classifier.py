from torch import nn
from torch.optim import Adam,SGD
from torch.utils.data import DataLoader
from torchvision import datasets
from  torchvision.transforms import ToTensor,Lambda,Compose

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=Compose([ToTensor(),Lambda(lambda x: x.view(-1))])
)
