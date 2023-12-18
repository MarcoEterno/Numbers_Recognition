import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, transforms
from PIL import Image

from config import get_system_device, batch_size, data_path, custom_data_path


def get_MNIST_data(train=True, data_loading=True):
    print(f"Loading the MNIST {'training' if train else 'test'} dataset")
    data = datasets.MNIST(
        root=data_path,
        train=train,
        download=True,
        transform=ToTensor()
    )
    if data_loading:
        dataset = DataLoader(data, batch_size=batch_size, shuffle=True)
        return dataset
    else:
        return data

def check_bounds_for_n_digits_dataset(n: int, train=True):
    if n < 1:
        raise ValueError("Expected number of digits for the dataset to create is greater than 1")
    if n > 9:
        raise ValueError("Expected number of digits for the dataset to create is less than 10.")


# create a dataset containing images of n digits from MNIST
def get_n_digits_dataset(n: int, train=True, augment_data=False, scale_data_linearly=False, device=get_system_device()):

    #check edge cases for n_digits_dataset
    check_bounds_for_n_digits_dataset(n, train)
    if n == 1:
        return get_MNIST_data(train=train, data_loading=True)

    dataset_type = 'train' if train else 'test'
    dataset_path = os.path.join(custom_data_path, f'{n}_digits_dataset_{dataset_type}.pt')
    # if n digits dataset already exists, load it
    if os.path.exists(dataset_path):
        print(f"Loading the {n} digits {dataset_type} dataset")
        n_digits_dataset = DataLoader(torch.load(dataset_path, map_location='cpu'), batch_size=batch_size, shuffle=True)
        return n_digits_dataset

    # if n digits dataset does not exist, create it
    tuples = []
    mnist_dataset = get_MNIST_data(train=train, data_loading=False)
    print(f"Creating the {n} digits {dataset_type} dataset")

    for idx in range(len(mnist_dataset)):
        # Get n consecutive images (wrapping around at the end)
        for i in range(n):
            img, label = mnist_dataset[(idx + i) % len(mnist_dataset)]
            img = transforms.ToPILImage()(img)
            # Combine images side by side
            if i == 0:
                combined_img = Image.new('L', (28 * n, 28))
                combined_img.paste(img, (0, 0))
                combined_label = label
            else:
                combined_img.paste(img, (28 * i, 0))
                combined_label = 10 * combined_label + label

        # Convert combined image and label back to tensor
        combined_img = transforms.ToTensor()(combined_img).to(device)
        combined_label = torch.tensor(combined_label, dtype=torch.long).to(device)

        tuples.append((combined_img, combined_label))

    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    print(f"{n} digits {dataset_type} dataset created. Saving dataset to {dataset_path}")
    torch.save(tuples, dataset_path)
    print("Loading the dataset")
    n_digits_dataset = DataLoader(tuples, batch_size=batch_size, shuffle=True)
    return n_digits_dataset


def augment_data(dataset: DataLoader, image_upsizing=1):
    """
    data_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomResizedCrop(224),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])"""
    augmented_dataset = []
    for img, label in dataset:
        for i in range(image_upsizing):
            img = transforms.ToPILImage()(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.RandomRotation(20)(img)
            img = transforms.RandomResizedCrop(224)(img)
            img = transforms.RandomVerticalFlip()(img)
            img = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)(img)
            img = transforms.ToTensor()(img).to(device)
            augmented_dataset.append((img, label))
    return DataLoader(augmented_dataset, batch_size=batch_size, shuffle=True)


if __name__ == '__main__':
    single_digits_train = get_MNIST_data(train=True)
    print(next(iter(single_digits_train))[0][0].shape)
    two_digits_train = get_n_digits_dataset(train=True)
    print(next(iter(two_digits_train))[0][0].shape)
