import torch
import torch.nn as nn
import torch.nn.functional as F

# defines a 4 layer autoencoder
class Autoencoder_4L (nn.Module):
    def __init__(self, base_channels=32, second_channels=64, third_channels=128, latent_dim=64):
        super (Autoencoder_4L, self).__init__()
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

# defines a 3 layer autoencoder
class Autoencoder_3L (nn.Module):
    def __init__(self, base_channels=32, second_channels=64, latent_dim=32):
        super (Autoencoder_3L, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, base_channels, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, second_channels, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(second_channels, latent_dim, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, second_channels, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(second_channels, base_channels, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(base_channels, 1, 3, stride=2, padding=1, output_padding=1)
        )
    def forward (self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# defines a 2 layer autoencoder
class Autoencoder_2L (nn.Module):
    def __init__(self, base_channels=32, latent_dim=16):
        super (Autoencoder_2L, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, base_channels, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, latent_dim, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, base_channels, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(base_channels, 1, 3, stride=2, padding=1, output_padding=1)
        )
    def forward (self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    