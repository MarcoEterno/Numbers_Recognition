from torch import nn, save, load
from torch.optim import Adam



class ImageClassifier(nn.Module):
    def __init__(self, loss_fn=nn.CrossEntropyLoss, optimizer=Adam, lr=1e-3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, ),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, ),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 22 * 22, 10)  # 22x22 is the size of the image after the convolutions
        )
        if optimizer == Adam:
            self.optimizer = Adam(self.parameters(), lr=lr)
        else:
            self.optimizer = optimizer(self.parameters(), lr=lr)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)