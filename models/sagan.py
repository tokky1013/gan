import torch.nn as nn
from models.layers import SelfAttention

class Generator(nn.Module):
    def __init__(self, n_channel=100, bias=False):
        super(Generator, self).__init__()

        self.layers = nn.ModuleDict({
            'layer0':nn.Sequential(
                nn.utils.spectral_norm(nn.ConvTranspose2d(n_channel, 512, 4, 1, 0, bias=bias)),
                nn.BatchNorm2d(512),
                nn.ReLU()
            ),
            'layer1':nn.Sequential(
                nn.utils.spectral_norm(nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=bias)),
                nn.BatchNorm2d(256),
                nn.ReLU()
            ),
            'layer2':nn.Sequential(
                nn.utils.spectral_norm(nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=bias)),
                nn.BatchNorm2d(128),
                nn.ReLU()
            ),
            'layer3':nn.Sequential(
                SelfAttention(128)
            ),
            'layer4':nn.Sequential(
                nn.utils.spectral_norm(nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=bias)),
                nn.BatchNorm2d(64),
                nn.ReLU()
            ),
            'layer5':nn.Sequential(
                SelfAttention(64)
            ),
            'layer6':nn.Sequential(
                nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=bias),
                nn.Tanh()
            )
        })

    def forward(self, z):
        for layer in self.layers.values():
            z = layer(z)
        return z.squeeze()

class Discriminator(nn.Module):
    def __init__(self, bias=False):
        super(Discriminator, self).__init__()

        self.layers = nn.ModuleDict({
            'layer0':nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(3, 64, 4, 2, 1, bias=bias)),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            'layer1':nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(64, 128, 4, 2, 1, bias=bias)),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            'layer2':nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(128, 256, 4, 2, 1, bias=bias)),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            'layer3':nn.Sequential(
                SelfAttention(256)
            ),
            'layer4':nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(256, 512, 4, 2, 1, bias=bias)),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            'layer5':nn.Sequential(
                SelfAttention(512)
            ),
            'layer6':nn.Sequential(
                nn.Conv2d(512, 1, 4, 1, 0, bias=bias),
            )
        })

    def forward(self, x):
        for layer in self.layers.values():
            x = layer(x)
        return x.squeeze()