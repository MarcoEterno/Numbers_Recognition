import math

from torch import nn, save, load
from torch.optim import Adam

from config import fast_training

import numpy as np

class ImageClassifier(nn.Module):
    def __init__(self, n_digits_to_recognize=1, image_upsizing=1, loss_fn=nn.CrossEntropyLoss, optimizer=Adam, lr=1e-3,
                 print_info=True):
        super().__init__()
        self.numbers_to_recognize = n_digits_to_recognize
        self.image_upsizing = image_upsizing
        self.model = None
        if n_digits_to_recognize > 4:
            pass
            # raise NotImplementedError(
            # "Too many digits to recognize. This model can only recognize up to 4 digits. "
            # "If you feel brave and want to set your machine on fire, remove this safety check and try again.")
        if n_digits_to_recognize < 1:
            raise ValueError("Number of digits to recognize must be greater than 0.")
        if image_upsizing < 1:
            raise ValueError("Image upsizing factor must be greater than 1.")
        if fast_training:
            self.initialize_fast_training_model()
        else:
            self.initialize_high_accuracy_model()

        if optimizer == Adam:
            self.optimizer = Adam(self.parameters(), lr=lr)
        else:
            self.optimizer = optimizer(self.parameters(), lr=lr)
        self.loss = loss_fn()
        if print_info:
            print(self)

    def forward(self, x):
        return self.model(x)

    def initialize_fast_training_model(self):
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

            nn.AdaptiveAvgPool2d((1, 1)),

            nn.Flatten(),
            nn.Linear(in_features=32,
                      out_features=10 * self.numbers_to_recognize * 4),
            nn.ReLU(),
            nn.Linear(in_features=10 * self.numbers_to_recognize * 4,
                      out_features=10 * self.numbers_to_recognize * 4),
            nn.ReLU(),
            # in_features = in_channels*dim_input // 4**(n_max_pooling_layers)
            nn.Linear(in_features=10 * self.numbers_to_recognize * 4, out_features=10 ** self.numbers_to_recognize)
        )

    # model accuracy on test dataset is:  0.9765625, epoch time = 7.154s, n_parameters = 42.7k
    # def initialize_fast_training_model(self):
    #     self.model = nn.Sequential(
    #         nn.Conv2d(1, 32, 3, padding=1),
    #         nn.BatchNorm2d(32),
    #         nn.ReLU(),
    #         nn.MaxPool2d(kernel_size=2),
    #         nn.Dropout(0.25),
    #
    #         nn.Conv2d(32, 64, 3, padding=1),
    #         nn.BatchNorm2d(64),
    #         nn.ReLU(),
    #         nn.MaxPool2d(kernel_size=2),
    #         nn.Dropout(0.25),
    #
    #         nn.Conv2d(64, 32, 3, padding=1),
    #         nn.BatchNorm2d(32),
    #         nn.ReLU(),
    #         nn.MaxPool2d(kernel_size=2),
    #         nn.Dropout(0.25),
    #
    #         nn.AdaptiveAvgPool2d((1, 1)),
    #
    #         nn.Flatten(),
    #         nn.Linear(in_features=32,
    #                   out_features=10 * 2 * self.numbers_to_recognize),
    #         nn.ReLU(),
    #         # in_features = in_channels*dim_input // 4**(n_max_pooling_layers)
    #         nn.Linear(in_features=10 * 2 * self.numbers_to_recognize, out_features=10 ** self.numbers_to_recognize)
    #     )



    def initialize_high_accuracy_model(self):
        print(
            "High accuracy mode is not reccomended and only works for 3 digits. Please use fast training mode instead.")
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.AdaptiveAvgPool2d((1, 1)),

            nn.Flatten(),
            nn.Linear(in_features=256,
                      out_features=2*10 * self.numbers_to_recognize),
            nn.ReLU(),
            # in_features = in_channels*dim_input // 4**(n_max_pooling_layers)
            nn.Linear(in_features=2*10 * self.numbers_to_recognize, out_features=10 ** self.numbers_to_recognize),
        )
