from torch import nn, save, load
from torch.optim import Adam


class ImageClassifier(nn.Module):
    def __init__(self, n_digits_to_recognize=1, image_upsizing=1 ,loss_fn=nn.CrossEntropyLoss, optimizer=Adam, lr=1e-3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  # padding is added to keep the size of the image constant
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            # nn.Conv2d(64, 64, 3, ),
            # nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 28 * 28*n_digits_to_recognize*image_upsizing, 10)
            # nn.Linear(64 * 22 * 22, 10)
        )
        print(self)
        if optimizer == Adam:
            self.optimizer = Adam(self.parameters(), lr=lr)
        else:
            self.optimizer = optimizer(self.parameters(), lr=lr)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)
