#動作未確認
import torch.nn as nn
from models.layers import ResidualBlock

class Generator(nn.Module):
    def __init__(self, res_block):
        super(Generator, self).__init__()
        self.encode_block = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, kernel_size=7, stride=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        )
        res_blocks = [ResidualBlock(256) for _ in range(res_block)]
        self.res_block = nn.Sequential(
            *res_blocks
        )
        self.decode_block = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, kernel_size=7, stride=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.encode_block(x)
        x = self.res_block(x)
        x = self.decode_block(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.layers = nn.ModuleDict({
            'layer0':nn.Sequential(
                nn.Conv2d(3, 64, 4, 2, 1),
                nn.InstanceNorm2d(64),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            'layer1':nn.Sequential(
                nn.Conv2d(64, 128, 4, 2, 1, bias=True),
                nn.InstanceNorm2d(128),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            'layer2':nn.Sequential(
                nn.Conv2d(128, 256, 4, 2, 1, bias=True),
                nn.InstanceNorm2d(256),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            'layer3':nn.Sequential(
                nn.Conv2d(256, 512, 4, 1, 1, bias=True),
                nn.InstanceNorm2d(512),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            'layer4':nn.Sequential(
                nn.Conv2d(512, 1, 4, 1, 1, bias=True),
            )
        })

    def forward(self, x):
        for layer in self.layers.values():
            x = layer(x)
        return x.squeeze()

#(n+2p-k)/s + 1
#n-12