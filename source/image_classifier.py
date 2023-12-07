import torch
from torch import nn, save, load
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
from torch.utils.tensorboard import SummaryWriter


def get_system_device():
    if torch.has_mps:
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def get_MNIST_data():
    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    dataset = DataLoader(training_data, batch_size=32, shuffle=True)
    return dataset


# images are 28x28 pixels, and have 10 classes (digits)

class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, ),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, ),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 22 * 22, 10)
        )

    def forward(self, x):
        return self.model(x)


# Training loop
def train_loop(datasets, model, loss_fn, optimizer, device, epochs=10):
    # Create a SummaryWriter instance. To access TensorBoard, run the following command in terminal
    # tensorboard --logdir=runs
    writer = SummaryWriter()  # logdir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    for epoch in range(epochs):
        # used to calculate the accuracy
        total_predictions = 0
        correct_predictions = 0

        for batch in datasets:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            output = clf(x)
            loss = loss_fn(output, y)

            # Backpropagation
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Calculate accuracy
            _, predicted = torch.max(output.data, 1)
            total_predictions += y.size(0)
            correct_predictions += (predicted == y).sum().item()

            # Log loss to tensorboard
            writer.add_scalar("Loss/train", loss.item(), epoch)

        # Log accuracy to tensorboard
        writer.add_scalar("Accuracy/train", (output.argmax(1) == y).sum().item() / len(y), epoch)
        print(f"Epoch: {epoch} - Loss:  {loss.item()}")

    # Close the writer instance
    writer.flush()
    writer.close()


if __name__ == '__main__':
    device = get_system_device()
    print(f"Using {device} device")

    datasets = get_MNIST_data()

    clf = ImageClassifier().to(device)
    print(clf)
    opt = Adam(clf.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    #train
    train_loop(datasets, clf, loss_fn, opt, device, epochs=15)

    # Save model
    with open("model_state_15.pt", "wb") as f:
        save(clf.state_dict(), f)
    # Load model
    with open("model_state_15.pt", "rb") as f:
        clf.load_state_dict(load(f))

"""
    # Test model
    clf.eval()
    x, y = next(iter(datasets))
    x = x.to(device)
    y = y.to(device)
    output = clf(x)
    print(output.argmax(1))
    print(y)
    print((output.argmax(1) == y).sum().item() / len(y))"""
