import torch
import torch.nn as nn


class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=5
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=5
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )
        )

        self.classifier = nn.Sequential(

            nn.Linear(64 * 5 * 5, 120),

            nn.ReLU(),

            nn.Linear(120, 84),

            nn.ReLU(),

            nn.Linear(84, 10)
        )

    def forward(self, x):

        x = self.features(x)

        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x