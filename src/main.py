"""Fichier principal du projet Variational-Autoencoder-Implementation.

Ce fichier est responsable de l'initialisation de l'environnement de travail, de la lecture de la configuration
et de l'appel des fonctions d'entraînement et de test.

Auteur: [Votre nom]
Date: [La date d'aujourd'hui]
"""

import os
import json
import argparse
from src.utils import load_config
from src.vae import VariationalAutoencoder
from src.datasets import load_dataset
from src.train import train_model
from src.test import run_tests

def main():
    """Fonction principale du programme.

    Cette fonction est responsable de l'initialisation de l'environnement de travail, de la lecture de la configuration
    et de l'appel des fonctions d'entraînement et de test.
    """
    # Parse les arguments de ligne de commande
    parser = argparse.ArgumentParser(description="Variational-Autoencoder-Implementation")
    parser.add_argument("--config", help="Fichier de configuration du modèle", default="src/config.json")
    parser.add_argument("--mode", help="Mode d'exécution (train/test)", choices=["train", "test"], default="train")
    args = parser.parse_args()

    # Lire la configuration du modèle
    config = load_config(args.config)

    # Charger les données d'entraînement et de test
    train_dataset, test_dataset = load_dataset(config["dataset"])

    # Créer un objet VariationalAutoencoder
    vae = VariationalAutoencoder(config)

    # Appeler la fonction d'entraînement ou de test en fonction du mode d'exécution
    if args.mode == "train":
        train_model(vae, train_dataset, test_dataset, config)
    elif args.mode == "test":
        run_tests(vae, test_dataset, config)

if __name__ == "__main__":
    main()
```

```python
# src/utils.py

import json

def load_config(config_file):
    """Lire la configuration du modèle à partir d'un fichier JSON.

    Args:
        config_file (str): Chemin du fichier de configuration.

    Returns:
        dict: Dictionnaire contenant la configuration du modèle.
    """
    with open(config_file, "r") as f:
        return json.load(f)