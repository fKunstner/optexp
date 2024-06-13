from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from optexp.config import get_logger
from optexp.models.model import Model


@dataclass
class SimpleMNISTCNN(Model):
    def load_model(self, input_shape, output_shape):
        get_logger().info("Creating model: " + self.__class__.__name__)
        return SimpleMNISTCNNTorch(input_shape[0], output_shape[0])


class SimpleMNISTCNNTorch(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 2 * 16, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(2 * 16, 2 * 50, 5, padding=2)
        self.fc1 = nn.Linear(2 * 50 * 7 * 7, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.fc1(x)
        return x


@dataclass
class SimpleImageNetCNN(Model):
    def load_model(self, input_shape, output_shape):
        get_logger().info("Creating model: " + self.__class__.__name__)
        return SimpleImageNetCNNTorch(input_shape[0], output_shape[0])


class SimpleImageNetCNNTorch(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 5, padding=2, stride=(2, 2))
        self.conv2 = nn.Conv2d(64, 64, 5, padding=2)
        self.conv3 = nn.Conv2d(64, 64, 5, padding=2)
        self.conv4 = nn.Conv2d(64, 64, 5, padding=2)
        self.conv5 = nn.Conv2d(64, 64, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(576, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)) + x)
        x = self.pool(F.relu(self.conv3(x)) + x)
        x = self.pool(F.relu(self.conv4(x)) + x)
        x = self.pool(F.relu(self.conv5(x)) + x)

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.fc1(x)
        return x


@dataclass
class ResNet50(Model):
    def load_model(self, input_shape, output_shape):
        get_logger().info("Creating model: " + self.__class__.__name__)
        return torchvision.models.resnet50(num_classes=output_shape[0])


@dataclass
class ResNet34(Model):
    def load_model(self, input_shape, output_shape):
        get_logger().info("Creating model: " + self.__class__.__name__)
        return torchvision.models.resnet34(num_classes=output_shape[0])


@dataclass
class ResNet34LayerNorm(Model):
    def load_model(self, input_shape, output_shape):
        get_logger().info("Creating model: " + self.__class__.__name__)
        return torchvision.models.resnet34(
            num_classes=output_shape[0], norm_layer=lambda x: nn.GroupNorm(1, x)
        )


@dataclass
class ResNet18(Model):
    def load_model(self, input_shape, output_shape):
        get_logger().info("Creating model: " + self.__class__.__name__)
        return torchvision.models.resnet18(num_classes=output_shape[0])


@dataclass
class ResNet18LayerNorm(Model):
    def load_model(self, input_shape, output_shape):
        get_logger().info("Creating model: " + self.__class__.__name__)
        return torchvision.models.resnet18(
            num_classes=output_shape[0], norm_layer=lambda x: nn.GroupNorm(1, x)
        )
