from torch import nn, save, load
from torch.optim import Adam

from config import fast_training


class ImageClassifier(nn.Module):
    def __init__(self, n_digits_to_recognize=1, image_upsizing=1, loss_fn=nn.CrossEntropyLoss, optimizer=Adam, lr=1e-3):
        super().__init__()
        self.model = None
        if n_digits_to_recognize > 9:
            raise NotImplementedError("Too many digits to recognize. This model can only recognize up to 9 digits.")
        if n_digits_to_recognize < 1:
            raise ValueError("Number of digits to recognize must be greater than 0.")
        if image_upsizing < 1:
            raise ValueError("Image upsizing factor must be greater than 1.")
        if fast_training:
            self.fast_training_model(n_digits_to_recognize, image_upsizing)
        else:
            self.high_accuracy_model(n_digits_to_recognize, image_upsizing)

        if optimizer == Adam:
            self.optimizer = Adam(self.parameters(), lr=lr)
        else:
            self.optimizer = optimizer(self.parameters(), lr=lr)
        self.loss = loss_fn()
        self.numbers_to_recognize = n_digits_to_recognize
        print(self)

    def forward(self, x):
        return self.model(x)

    def fast_training_model(self, n_digits_to_recognize=1, image_upsizing=1):
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(in_features=3136 * n_digits_to_recognize,
                      out_features=10 ** n_digits_to_recognize),  # 3136*n_digits_to_recognize |
            # for high perf  1280 * n_digits_to_recognize
            nn.Linear(in_features=10 ** n_digits_to_recognize, out_features=10 ** n_digits_to_recognize)
        )

    def high_accuracy_model(self, n_digits_to_recognize=1, image_upsizing=1):
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),

            nn.Flatten(),
            nn.Linear(in_features=1280 * n_digits_to_recognize,
                      out_features=10 ** n_digits_to_recognize),
            nn.Linear(in_features=10 ** n_digits_to_recognize, out_features=10 ** n_digits_to_recognize)
        )
