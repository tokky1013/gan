import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """Self-attention GANにおけるSelf-attention
    ただしSAGANのSpectral Normを抜いているので注意

    Arguments:
        dims {int} -- 4Dテンソルの入力チャンネル
    """
    def __init__(self, dims):
        super().__init__()
        self.conv_theta = nn.Conv2d(dims, dims // 8, kernel_size=1)
        self.conv_phi = nn.Conv2d(dims, dims // 8, kernel_size=1)
        self.conv_g = nn.Conv2d(dims, dims // 2, kernel_size=1)
        self.conv_attn = nn.Conv2d(dims // 2, dims, kernel_size=1)
        self.sigma_ratio = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, inputs):
        batch, ch, height, width = inputs.size()
        # theta path
        theta = self.conv_theta(inputs)
        theta = theta.view(batch, ch // 8, height * width).permute([0, 2, 1])  # (B, HW, C/8)        
        # phi path
        phi = self.conv_phi(inputs)
        phi = F.max_pool2d(phi, kernel_size=2)  # (B, C/8, H/2, W/2)
        phi = phi.view(batch, ch // 8, height * width // 4)  # (B, C/8, HW/4)
        # attention
        attn = torch.bmm(theta, phi)  # (B, HW, HW/4)
        attn = F.softmax(attn, dim=-1)
        # g path
        g = self.conv_g(inputs)
        g = F.max_pool2d(g, kernel_size=2)  # (B, C/2, H/2, W/2)
        g = g.view(batch, ch // 2, height * width // 4).permute([0, 2, 1])  # (B, HW/4, C/2)

        attn_g = torch.bmm(attn, g)  # (B, HW, C/2)
        attn_g = attn_g.permute([0, 2, 1]).view(batch, ch // 2, height, width)  # (B, C/2, H, W)
        attn_g = self.conv_attn(attn_g)
        return inputs + self.sigma_ratio * attn_g

class ResidualBlock(nn.Module):
    def __init__(self, input_channel):
        super(ResidualBlock, self).__init__()

        self.residualblock = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(input_channel, input_channel, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(input_channel),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(input_channel, input_channel, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(input_channel))

    def forward(self, x):
        residual = self.residualblock(x)

        return x + residual