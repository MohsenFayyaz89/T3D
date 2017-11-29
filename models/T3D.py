import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

__all__ = ['DenseNet', 'densenet121', 'densenet161'] # with DropOut

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm.1', nn.BatchNorm3d(num_input_features)),
        self.add_module('relu.1', nn.ReLU(inplace=True)),
        self.add_module('conv.1', nn.Conv3d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm.2', nn.BatchNorm3d(bn_size * growth_rate)),
        self.add_module('relu.2', nn.ReLU(inplace=True)),
        self.add_module('conv.2', nn.Conv3d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))

class _TTL(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()

        self.b1 = nn.Sequential(
            self.add_module('norm', nn.BatchNorm3d(num_input_features))
            self.add_module('relu', nn.ReLU(inplace=True))
            self.add_module('conv', nn.Conv3d(num_input_features, 128,
                                          kernel_size=1, stride=1, bias=False))
        )

        self.b2 = nn.Sequential(
            self.add_module('norm', nn.BatchNorm3d(num_input_features))
            self.add_module('relu', nn.ReLU(inplace=True))
            self.add_module('conv', nn.Conv3d(num_input_features, 128,
                                          kernel_size=(3,3,3), stride=1, bias=False))
        )

        self.b3 = nn.Sequential(
            self.add_module('norm', nn.BatchNorm3d(num_input_features))
            self.add_module('relu', nn.ReLU(inplace=True))
            self.add_module('conv', nn.Conv3d(num_input_features, 128,
                                          kernel_size=(4,3,3), stride=1, bias=False))
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        return torch.cat([y1,y2,y3], 1)



class DenseNet3D(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        super(DenseNet3D, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(3, num_init_features, kernel_size=(3, 7, 7), stride=2, padding=(1, 3, 3), bias=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            
            if i != len(block_config) - 1:
                ttl = _TTL(num_input_features=num_features)
                self.features.add_module('ttl%d' % (i + 1), ttl)
                num_features = 128*3

                trans = _Transition(num_input_features=num_features, num_output_features=num_features )
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool3d(out, kernel_size=(1,7,7)).view(features.size(0), -1)
        out = self.classifier(out)
        return out


def densenet121_3D():
    model = DenseNet3D(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16))
    return model


def densenet121_3D_DropOut():
    model = DenseNet3D(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), drop_rate=0.2)
    return model


def densenet169_3D():
    model = DenseNet3D(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32))
    return model



def densenet161_3D():
    model = DenseNet3D(num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24))
    return model



def densenet161_3D_DropOut():
    model = DenseNet3D(num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24), drop_rate=0.2)
    return model


def densenet201_3D():
    model = DenseNet3D(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32))
    return model


def densenet121(**kwargs):
    """Constructs a DenseNet-121_DropOut model.
    """
    model = DenseNet3D(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), drop_rate=0.2, **kwargs)
    return model


def densenet161(**kwargs):
    """Constructs a DenseNet-161_DropOut model.
    """
    model = DenseNet3D(num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24), drop_rate=0.2, **kwargs)
    return model

