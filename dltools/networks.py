import torch
import torch.nn as nn


class LinearNormRelu(nn.Module):
    """Linear (fully connected) Normalization Relu block
    """
    def __init__(self, inc, outc, relu=True, norm=None):
        super().__init__()
        self.linear = nn.Linear(inc, outc)
        self.relu = relu
        self.norm = norm

        if norm is None:
            self.norm_layer = None
        elif norm == "bn":
            self.norm_layer = nn.BatchNorm1d(outc)
        elif norm == "in":
            self.norm_layer = nn.InstanceNorm1d(outc)

    def forward(self, x):
        out = self.linear(x)
        if self.norm is not None:
            out = self.norm_layer(out)
        if self.relu:
            out = torch.relu(out)

        return out


class LinearBlock(nn.Module):
    """A sequence of LinearNormRelu blocks
    """
    def __init__(self, inc, outc, itmc, L, relu=False, norm=None):
        super().__init__()
        layers = []
        layers.append(LinearNormRelu(inc, itmc, norm=norm))

        for i in range(L - 2):
            layers.append(LinearNormRelu(itmc, itmc, norm=norm))
        layers.append(LinearNormRelu(itmc, outc, relu=relu, norm=norm))

        self.layers = nn.ModuleList(layers)
        self.L = L

    def forward(self, x):
        out = x
        for i in range(self.L):
            out = self.layers[i](out)

        return out


class ConvNormRelu(nn.Module):
    """Comvolution block containing a nn.Conv2d, batchnormalization and relu layer
    """
    def __init__(self, inc, outc, k=3, stride=1, relu=True, norm="bn", D=2):
        """Instance initialization

        Args:
            inc (int): number of input channels
            outc (int): number of output channels
            stride (int, optional): stride for convolution. Defaults to 1.
            relu (bool, optional): whether to use relu activation or not. Defaults to True.
            norm (string, optional): normalization to use between layers.
                                    Options between 'bn' for batchnormalization
                                    and 'None' for no normalization. Defaults to 'bn'.
        """
        super().__init__()

        if D == 2:
            self.conv = nn.Conv2d(inc, outc, k, padding=k // 2, stride=stride)
            if norm is None:
                self.norm_layer = None
            elif norm == "bn":
                self.norm_layer = nn.BatchNorm2d(outc)
            elif norm == "in":
                self.norm_layer = nn.InstanceNorm2d(outc)
        elif D == 3:
            self.conv = nn.Conv3d(inc, outc, k, padding=k // 2, stride=stride)
            if norm is None:
                self.norm_layer = None
            elif norm == "bn":
                self.norm_layer = nn.BatchNorm3d(outc)
            elif norm == "in":
                self.norm_layer = nn.InstanceNorm3d(outc)

        self.relu = relu
        self.norm = norm
        self.D = D

    def forward(self, x):
        out = self.conv(x)
        if self.norm is not None:
            out = self.norm_layer(out)
        if self.relu:
            out = torch.relu(out)

        return out


class ConvBlock(nn.Module):
    """Comvolution block containing a sequence of ConvNormRelu blocks.
    The stride of the last block is given by stride parameter
    """
    def __init__(self, inc, outc, itmc, L, k=3, stride=1, relu=True, norm="bn", D=2):
        """Instance initialization

        Args:
            inc (int): number of input channels
            outc (int): number of output channels
            itmc (int): number of channels for the hidden layers
            L (int): number of ConvNormRelu blocks in sequence
            stride (int, optional): Stride of the last ConvNormRelu layer. Defaults to 1.
            norm (string, optional): normalization to use between layers.
                                    Options between 'bn' for batchnormalization
                                    and 'ln' for layer normalization. Defaults to 'bn'.
        """
        super().__init__()

        layers = []
        layers.append(ConvNormRelu(inc, itmc, k=k, norm=norm, D=D))

        for i in range(L - 2):
            layers.append(ConvNormRelu(itmc, itmc, k=k, norm=norm, D=D))
        layers.append(ConvNormRelu(itmc, outc, k=k, stride=stride, norm=norm, relu=relu, D=D))

        self.layers = nn.ModuleList(layers)
        self.L = L

    def forward(self, x):
        out = x
        for i in range(self.L):
            out = self.layers[i](out)

        return out
