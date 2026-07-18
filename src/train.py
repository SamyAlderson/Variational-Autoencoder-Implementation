"""
Fichier src/train.py

Contient la fonction d'entraînement du modèle Variational Autoencoder.

Author: [Votre nom]
Date: [Date de création]
"""

import os
import argparse
import json
import time
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from src.config import Config
from src.vae import VAE
from src.datasets import get_dataset
from src.utils import set_seed, get_device

def get_args():
    """
    Récupère les arguments de ligne de commande.
    """
    parser = argparse.ArgumentParser(description='Variational Autoencoder Implementation')
    parser.add_argument('--config', type=str, required=True, help='Fichier de configuration du modèle')
    parser.add_argument('--epochs', type=int, default=100, help='Nombre d'épochs d\'entraînement')
    parser.add_argument('--batch_size', type=int, default=32, help='Taille du batch')
    parser.add_argument('--lr', type=float, default=0.001, help='Taux d\'apprentissage')
    parser.add_argument('--save_model', action='store_true', help='Enregistrer le modèle entièrement entraîné')
    return parser.parse_args()

def train(model, device, train_loader, optimizer, epoch):
    """
    Étend le modèle sur un lot d\'entraînement.

    Args:
        model (VAE): Modèle VAE
        device (torch.device): Dispositif sur lequel s\'exécute le modèle
        train_loader (DataLoader): Chargement des données d\'entraînement
        optimizer (Optimizer): Optimiseur utilisé pour l\'entraînement
        epoch (int): Numéro de l\'étape actuelle
    """
    model.train()
    total_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_loss, kl_loss = model(data)
        loss = recon_loss + kl_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    logging.info(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}')

def main():
    """
    Fonction principale d\'entraînement du modèle.
    """
    args = get_args()
    config = Config(args.config)
    device = get_device()
    model = VAE(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train_dataset = get_dataset('train', config)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    for epoch in range(args.epochs):
        train(model, device, train_loader, optimizer, epoch)
        if args.save_model:
            torch.save(model.state_dict(), os.path.join(config.model_dir, 'model.pth'))
    logging.info('Entraînement terminé.')

if __name__ == '__main__':
    set_seed(42)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    main()
```

```python
# src/utils.py

import torch
import random

def set_seed(seed):
    """
    Définit le seed pour la génération de nombres aléatoires.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    """
    Retourne le dispositif sur lequel s\'exécute le modèle.
    """
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    else:
        return torch.device('cpu')
```

```python
# src/config.py

import json

class Config:
    def __init__(self, config_file):
        """
        Initialise le fichier de configuration.

        Args:
            config_file (str): Chemin du fichier de configuration
        """
        with open(config_file, 'r') as f:
            self.config = json.load(f)

    def __getattr__(self, name):
        """
        Retourne la valeur d'une clé du fichier de configuration.

        Args:
            name (str): Nom de la clé
        """
        return self.config[name]
```

```python
# src/datasets.py

import os
import pandas as pd
from torchvision import datasets, transforms

def get_dataset(name, config):
    """
    Retourne le jeu de données spécifié.

    Args:
        name (str): Nom du jeu de données
        config (Config): Fichier de configuration
    """
    if name == 'train':
        data_dir = os.path.join(config.data_dir, 'train')
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    elif name == 'test':
        data_dir = os.path.join(config.data_dir, 'test')
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    return dataset