import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, transforms
from PIL import Image
import os

from config import get_system_device


def get_MNIST_data(train=True):
    data = datasets.MNIST(
        root="data",
        train=train,
        download=True,
        transform=ToTensor()
    )
    dataset = DataLoader(data, batch_size=256, shuffle=True)
    return dataset


# create a dataset containing images of two digits from MNIST
def create_n_digit_dataset(n, train=True, device=get_system_device()):

    return None


def get_MNIST_couples(train=True, device= get_system_device()):
    dataset_path = os.path.join(os.getcwd(), 'data','custom','two_digits_dataset.pt')
    print(dataset_path)
    if os.path.exists(dataset_path):
        print("loading the two digit dataset")
        two_digits_dataset = DataLoader(torch.load(dataset_path), batch_size=256, shuffle=True)
        return two_digits_dataset

    #if dataset does not exist, create it
    couples = []
    mnist_dataset = datasets.MNIST(
        root="data",
        train=train,
        download=True,
        transform=ToTensor()
    )
    print("creating the two digit dataset")
    for idx in range(len(mnist_dataset)):
        # Get two consecutive images (wrapping around at the end)
        img1, label1 = mnist_dataset[idx]
        img2, label2 = mnist_dataset[(idx + 1) % len(mnist_dataset)]

        # Convert tensors to PIL images for concatenation
        img1 = transforms.ToPILImage()(img1)
        img2 = transforms.ToPILImage()(img2)

        # Combine images side by side
        combined_img = Image.new('L', (56, 28))
        combined_img.paste(img1, (0, 0))
        combined_img.paste(img2, (28, 0))

        # Combine labels into a string
        combined_label = 10* label1 + label2

        # Convert combined image and label back to tensor
        combined_img = transforms.ToTensor()(combined_img).to(device)
        combined_label = torch.tensor(combined_label, dtype=torch.long).to(device)

        couples.append((combined_img, combined_label))

    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    print(f"Two digits dataset created. Saving dataset to {dataset_path}")
    torch.save(couples, dataset_path)
    two_digits_dataset = DataLoader(couples, batch_size=256, shuffle=True)
    return two_digits_dataset


if __name__ == '__main__':
    single_digits_train= get_MNIST_data(train=True)
    print(next(iter(single_digits_train))[0][0].shape)
    two_digits_train = get_MNIST_couples(train=True)
    print(next(iter(two_digits_train))[0][0].shape)

