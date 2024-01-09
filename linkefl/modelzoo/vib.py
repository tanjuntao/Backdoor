import torch.nn.functional as F
import torch.random
from torch import nn


class DeepVIB(nn.Module):
    def __init__(self, input_shape, output_shape, z_dim):
        super(DeepVIB, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.z_dim = z_dim

        # build encoder
        intermediate_size = 64
        self.encoder = nn.Sequential(
            nn.Linear(input_shape, intermediate_size),
            nn.ReLU(inplace=True),
            nn.Linear(intermediate_size, intermediate_size),
            nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(intermediate_size, self.z_dim)
        self.fc_std = nn.Linear(intermediate_size, self.z_dim)

        # build decoder
        self.decoder = nn.Linear(self.z_dim, output_shape)

    def encode(self, x):
        x = self.encoder(x)
        return self.fc_mu(x), F.softplus(self.fc_std(x) - 5, beta=1)

    def decode(self, z):
        """
        z : [batch_size,z_dim]
        """
        return self.decoder(z)

    def reparameterise(self, mu, std):
        """
        mu : [batch_size,z_dim]
        std : [batch_size,z_dim]
        """
        # get epsilon from standard normal
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        """
        Forward pass

        Parameters:
        -----------
        x :
        """
        # flattent image
        x_flat = x.view(x.size(0), -1)
        mu, std = self.encode(x_flat)
        z = self.reparameterise(mu, std)
        return self.decode(z), mu, std
