from typing import Literal

import torch
import torch.nn as nn
import torchvision

from optexp.models.model import Model


class LeNet5(Model):
    def load_model(self, input_shape=(1,), output_shape=(10,)):
        model = torch.nn.Sequential(
            nn.Conv2d(input_shape[0], 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, output_shape[0]),
        )
        return model


class ResNet(Model):
    size: Literal[18, 34, 50, 101, 152]

    def load_model(self, input_shape, output_shape):
        resnetX = getattr(torchvision.models, f"resnet{self.size}")
        return resnetX(num_classes=output_shape[0])
