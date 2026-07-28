import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder (nn.Module):
    def __init__(self, base_channels=32, second_channels=64, third_channels=128, latent_dim=64):
        super (Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, base_channels, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, second_channels, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(second_channels, third_channels, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(third_channels, latent_dim, 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, third_channels, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(third_channels, second_channels, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(second_channels, base_channels, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(base_channels, 1, 3, stride=2, padding=1, output_padding=1)
        )
    def forward (self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    