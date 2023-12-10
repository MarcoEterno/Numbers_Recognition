from torch import nn, save, load
from torch.optim import Adam


class ImageClassifier(nn.Module):
    def __init__(self, n_digits_to_recognize=1, image_upsizing=1 ,loss_fn=nn.CrossEntropyLoss, optimizer=Adam, lr=1e-3):
        super().__init__()
        if n_digits_to_recognize == 1:
            self.model = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1),  # padding is added to keep the size of the image constant
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                # nn.Conv2d(64, 64, 3, padding=1),
                # nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * 28 * 28 * image_upsizing, 10)
            )
        elif n_digits_to_recognize == 2:
            self.model = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1),  # padding is added to keep the size of the image constant
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                #nn.Conv2d(64, 64, 3, padding=1),
                #nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * 28 * 28 * 2 * image_upsizing, 10 ** n_digits_to_recognize)
            )
            """
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
                nn.Linear(in_features=6272, out_features=10 * n_digits_to_recognize) #2688 & 12544
            )"""

        else:
            raise NotImplementedError("Only 1 or 2 digits recognition has been implemented for now")
        if optimizer == Adam:
            self.optimizer = Adam(self.parameters(), lr=lr)
        else:
            self.optimizer = optimizer(self.parameters(), lr=lr)
        self.loss = nn.CrossEntropyLoss()
        self.numbers_to_recognize = n_digits_to_recognize
        print(self)

    def forward(self, x):
        return self.model(x)
