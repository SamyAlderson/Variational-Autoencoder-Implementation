"""
Variational Autoencoder Implementation

Fichier contenant la définition de l'architecture du VAE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Encodeur du VAE

    Args:
    - input_dim (int): Dimension de l'entrée
    - latent_dim (int): Dimension de l'espace latent
    - hidden_dim (int): Dimension des couches cachées
    """

    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, latent_dim * 2)  # mu et sigma

    def forward(self, x):
        """
        Propagation avant

        Args:
        - x (Tensor): Entrée

        Returns:
        - z (Tensor): Variable latente
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        z = self.fc3(x)
        mu, log_var = z.chunk(2, dim=1)
        return mu, log_var


class Decoder(nn.Module):
    """
    Décocodeur du VAE

    Args:
    - input_dim (int): Dimension de l'entrée
    - latent_dim (int): Dimension de l'espace latent
    - hidden_dim (int): Dimension des couches cachées
    """

    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)

    def forward(self, z):
        """
        Propagation avant

        Args:
        - z (Tensor): Variable latente

        Returns:
        - x_reconstructed (Tensor): Représentation reconstruite
        """
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        x_reconstructed = torch.sigmoid(self.fc3(z))
        return x_reconstructed


class VAE(nn.Module):
    """
    Variational Autoencoder

    Args:
    - input_dim (int): Dimension de l'entrée
    - latent_dim (int): Dimension de l'espace latent
    - hidden_dim (int): Dimension des couches cachées
    """

    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim, hidden_dim)
        self.decoder = Decoder(input_dim, latent_dim, hidden_dim)

    def forward(self, x):
        """
        Propagation avant

        Args:
        - x (Tensor): Entrée

        Returns:
        - x_reconstructed (Tensor): Représentation reconstruite
        - z (Tensor): Variable latente
        """
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, z, mu, log_var

    def reparameterize(self, mu, log_var):
        """
        Réparameterisation de la variable latente

        Args:
        - mu (Tensor): Moyenne
        - log_var (Tensor): Log-écart-type

        Returns:
        - z (Tensor): Variable latente
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

def loss_function(reconstructed, original, mu, log_var):
    """
    Fonction de perte

    Args:
    - reconstructed (Tensor): Représentation reconstruite
    - original (Tensor): Entrée originale
    - mu (Tensor): Moyenne
    - log_var (Tensor): Log-écart-type

    Returns:
    - loss (Tensor): Perte
    """
    BCE = F.binary_cross_entropy(reconstructed, original, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu ** 2 - torch.exp(log_var))
    return BCE + KLD

def train(model, device, train_loader, optimizer, epoch):
    """
    Entraînement du modèle

    Args:
    - model (VAE): Modèle
    - device (str): Disque dur
    - train_loader (DataLoader): Chargement des données d'entraînement
    - optimizer (Optimizer): Optimiseur
    - epoch (int): Époque
    """
    model.train()
    for batch_idx, (data, _) in enumerate(train_loader):
        data, _ = data.to(device), _.to(device)
        optimizer.zero_grad()
        reconstructed, _, mu, log_var = model(data)
        loss = loss_function(reconstructed, data, mu, log_var)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    """
    Évaluation du modèle

    Args:
    - model (VAE): Modèle
    - device (str): Disque dur
    - test_loader (DataLoader): Chargement des données de test
    """
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data, _ = data.to(device), _.to(device)
            reconstructed, _, mu, log_var = model(data)
            loss = loss_function(reconstructed, data, mu, log_var)
            test_loss += loss.item()
    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}'.format(test_loss))