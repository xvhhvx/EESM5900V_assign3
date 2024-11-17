import torch
import torch.nn as nn
import torch.nn.functional as F

def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """Creates a transposed-convolutional layer, with optional batch normalization.
    """
    layers = []
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True, init_zero_weights=False):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    if init_zero_weights:
        conv_layer.weight.data = torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.001
    layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)
  
class DCGenerator(nn.Module):
    def __init__(self):
        super(DCGenerator, self).__init__()

        self.deconv1 = deconv(in_channels=100, out_channels=128, kernel_size=4, stride=1, padding=0)
        
        self.deconv2 = deconv(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        
        self.deconv3 = deconv(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        
        self.deconv4 = deconv(in_channels=32, out_channels=3, kernel_size=4, stride=2, padding=1, batch_norm=False)
    
    def forward(self, z):
        """Generates an image given a sample of random noise."""
        
        out = F.relu(self.deconv1(z))
        out = F.relu(self.deconv2(out))
        out = F.relu(self.deconv3(out))
        out = F.tanh(self.deconv4(out))
        return out

class DCDiscriminator(nn.Module):
    """Defines the architecture of the discriminator network.
       Note: Both discriminators D_X and D_Y have the same architecture in this assignment.
    """
    def __init__(self):
        super(DCDiscriminator, self).__init__()
        
        self.conv1 = conv(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=1)
        
        self.conv2 = conv(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)
        
        self.conv3 = conv(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        
        self.conv4 = conv(in_channels=128, out_channels=1, kernel_size=4, stride=2, padding=0, batch_norm = False)
        
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))

        out = self.conv4(out).squeeze()
        out = torch.sigmoid(out)
        return out