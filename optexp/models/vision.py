import math
from typing import Literal, Optional

import torch
import torch.nn.functional as F
import torchvision

from optexp.models.initiliazation import InitializationStrategy
from optexp.models.model import Model, assert_batch_sizes_match
from optexp.models.transformer import TransformerModule


def validate_image_data(input_shape, output_shape):
    if len(input_shape) != 4:
        raise ValueError(
            "Input shape must be 4D, [batch, channels, width, height]. "
            f"Got {len(input_shape)} dimensions."
        )
    if len(output_shape) != 2:
        raise ValueError(
            "Output shape must be 2D, [batch, num_classes]. "
            f"Got {len(output_shape)} dimensions."
        )


class LeNet5(Model):
    """A basic convolutional neural network for image classification from [LeCun1998]_.

    The model expects images of shape [batch, channels, 32, 32].
    If images are 28x28, the model will pad the images to 32x32.

    .. [LeCun1998] Gradient Based Learning Applied to Document Recognition.
       Yann LeCun, Leon Bottou, Yoshua Bengio, and Patrick Haffner.
       Proceedings of the IEEE, 86(11):2278-2324, 1998.
       `DOI: 10.1109/5.726791 <https://doi.org/10.1109/5.726791>`_
    """

    def load_model(self, input_shape, output_shape):
        validate_image_data(input_shape, output_shape)

        b_in, channels, _, _ = input_shape
        b_out, num_classes = output_shape
        assert_batch_sizes_match(b_in, b_out)

        class LeNet5Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(channels, 6, 5)
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
    """A deep convolutional neural network from [He2016]_.

    The model expects images of shape [batch, channels, 224, 224].

    .. [He2016] Deep Residual Learning for Image Recognition.
        Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
        Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2016.
        `DOI: 10.1109/CVPR.2016.90 <https://doi.org/10.1109/CVPR.2016.90>`_

    Args:
        size (int): size of the model. Must be one of [18, 34, 50, 101, 152].
    """

    size: Literal[18, 34, 50, 101, 152]

    def __post_init__(self):
        if self.size not in [18, 34, 50, 101, 152]:
            raise ValueError(
                f"Invalid ResNet size. Got {self.size}. "
                "Expected one of [18, 34, 50, 101, 152]."
            )

    def load_model(self, input_shape, output_shape):
        validate_image_data(input_shape, output_shape)
        resnet_x = getattr(torchvision.models, f"resnet{self.size}")
        return resnet_x(num_classes=output_shape[0])


class SimpleViT(Model):
    patch_height: int = 16
    patch_width: int = 16

    n_layers: int = 12
    n_head: int = 6
    d_model: int = 384
    d_mlp: Optional[int] = 1536
    p_residual_dropout: float = 0.0
    p_attention_dropout: float = 0.0
    p_embedding_dropout: float = 0.0
    is_autoregressive: bool = False
    initialization: Optional[InitializationStrategy] = None

    def load_model(
        self, input_shape: torch.Size, output_shape: torch.Size
    ) -> torch.nn.Module:
        b_in, channels, height, width = input_shape
        b_out, num_classes = output_shape
        assert_batch_sizes_match(b_in, b_out)
        assert (
            height % self.patch_height == 0 and width % self.patch_width == 0
        ), "Patch Dimensions must divide Image Dimensions"
        model = SimpleViTModule(
            image_height=height,
            image_width=width,
            image_channels=channels,
            patch_height=self.patch_height,
            patch_width=self.patch_width,
            n_class=num_classes,
            n_layers=self.n_layers,
            n_head=self.n_head,
            d_model=self.d_model,
            d_mlp=self.d_mlp,
            p_residual_dropout=self.p_residual_dropout,
            p_attention_dropout=self.p_attention_dropout,
            p_embedding_dropout=self.p_embedding_dropout,
            is_autoregressive=self.is_autoregressive,
        )
        if self.initialization is not None:
            model = self.initialization.initialize(model)
        return model


class SimpleViTModule(TransformerModule):

    def __init__(
        self,
        image_height: int,
        image_width: int,
        image_channels: int,
        patch_height: int,
        patch_width: int,
        n_class: int,
        n_layers: int,
        n_head: int,
        d_model: int,
        d_mlp: int,
        p_residual_dropout: float,
        p_attention_dropout: float,
        p_embedding_dropout: float,
        is_autoregressive: bool,
    ):
        super().__init__(
            (image_height // patch_height) * (image_width // patch_width),
            n_class,
            n_layers,
            n_head,
            d_model,
            d_mlp,
            p_residual_dropout,
            p_attention_dropout,
            p_embedding_dropout,
            is_autoregressive,
        )
        self.image_height = image_height
        self.image_width = image_width
        self.image_channels = image_channels
        self.patch_height = patch_height
        self.patch_width = patch_width

        positional_encodings = self.get_2d_positional_encodings(
            self.image_height // self.patch_height,
            self.patch_height // self.patch_width,
            self.d_model,
        )
        self.register_buffer("positional_encodings", positional_encodings)

    # Stolen from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py
    def get_2d_positional_encodings(self, n_height: int, n_width: int, d_model: int):
        y, x = torch.meshgrid(
            torch.arange(n_height), torch.arange(n_width), indexing="ij"
        )
        assert (
            d_model % 4
        ) == 0, "feature dimension must be multiple of 4 for sincos emb"
        omega = torch.arange(d_model // 4) / (d_model // 4 - 1)
        omega = 1.0 / (10000**omega)

        y = y.flatten()[:, None] * omega[None, :]
        x = x.flatten()[:, None] * omega[None, :]
        pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
        return pe

    def get_prediction_layer(self, d_model: int, n_class: int) -> torch.nn.Module:
        class MeanAndPredict(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(d_model, n_class)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = x.mean(dim=1)
                return self.linear(x)

        return MeanAndPredict()

    def get_embedding_layer(self, d_model: int, n_class: int) -> torch.nn.Module:
        """Assumes x has  dimensions [batch, channels, height, width"""
        patch_dim = self.patch_height * self.patch_width * self.image_channels

        class Patchify(torch.nn.Module):
            patch_size: int

            def __init__(self, patch_height: int, patch_width: int):
                super().__init__()
                self.patch_height = patch_height
                self.patch_width = patch_width

            # TODO this would be much  more readable with einops rearrange but that adds a dependency
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                b, c, img_h, img_w = x.shape

                n_patch_h = img_h // self.patch_height
                n_patch_w = img_w // self.patch_width
                x = x.view(
                    b, c, n_patch_h, self.patch_height, n_patch_w, self.patch_width
                )
                x = x.permute(0, 2, 4, 3, 5, 1)
                x = x.reshape(
                    b, n_patch_w * n_patch_h, self.patch_height * self.patch_width * c
                )
                return x

        embedder = torch.nn.Sequential(
            Patchify(self.patch_height, self.patch_width),
            torch.nn.LayerNorm(patch_dim),
            torch.nn.Linear(patch_dim, d_model),
            torch.nn.LayerNorm(d_model),
        )
        return embedder

    def get_attention_mask(self, sequence_length: int):
        return None
