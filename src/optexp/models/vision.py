from typing import Literal

import torch
import torch.nn.functional as F
import torchvision

from optexp.models.model import Model


def validate_image_data(input_shape, output_shape):
    if len(input_shape) != 4:
        raise ValueError(
            "Input shape must be 4D, [batch, channels, width, height]. "
            f"Got {len(input_shape)} dimensions."
        )
    if len(output_shape) != 1:
        raise ValueError(
            "Output shape must be 1D, [num_classes]. "
            f"Got {len(output_shape)} dimensions."
        )


class LeNet5(Model):
    def load_model(self, input_shape, output_shape):
        validate_image_data(input_shape, output_shape)

        _, input_channels, _, _ = input_shape
        num_classes = output_shape[0]

        class LeNet5Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(input_channels, 6, 5)
                self.conv2 = torch.nn.Conv2d(6, 16, 5)
                self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
                self.fc2 = torch.nn.Linear(120, 84)
                self.fc3 = torch.nn.Linear(84, num_classes)

            def forward(self, x):
                if x.shape[-2:] == (28, 28):
                    pad = torchvision.transforms.Pad(2, fill=0, padding_mode="constant")
                    x = pad(x)
                else:
                    if not x.shape[-2:] == (32, 32):
                        raise ValueError(
                            f"Input shape must be 28x28 or 32x32. Got {x.shape}"
                        )

                x = F.max_pool2d(F.relu(self.conv1(x)), 2)
                x = F.max_pool2d(F.relu(self.conv2(x)), 2)
                x = torch.flatten(x, 1)
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                output = self.fc3(x)

                return output

        return LeNet5Module()


class ResNet(Model):
    size: Literal[18, 34, 50, 101, 152]

    def load_model(self, input_shape, output_shape):
        validate_image_data(input_shape, output_shape)
        resnet_x = getattr(torchvision.models, f"resnet{self.size}")
        return resnet_x(num_classes=output_shape[0])
