# src/test.py

"""
Fichier contenant les tests unitaires pour le projet Variational-Autoencoder-Implementation
"""

import unittest
import json
from src import vae, datasets, train, utils
from src.config import config

class TestVae(unittest.TestCase):

    def test_vae_init(self):
        """
        Teste la méthode d'initialisation de l'architecture du VAE
        """
        vae_architecture = vae.VAE()
        self.assertIsNotNone(vae_architecture.encoder)
        self.assertIsNotNone(vae_architecture.decoder)
        self.assertEqual(vae_architecture.z_dim, config["vae"]["z_dim"])
        self.assertEqual(vae_architecture.h_dim, config["vae"]["h_dim"])

    def test_load_datasets(self):
        """
        Teste la méthode de chargement des données d'entraînement et de test
        """
        datasets_loader = datasets.Datasets(config["data"]["path"])
        self.assertIsNotNone(datasets_loader.train_data)
        self.assertIsNotNone(datasets_loader.test_data)

    def test_train_vae(self):
        """
        Teste la fonction d'entraînement du VAE
        """
        vae_architecture = vae.VAE()
        train_vae = train.TrainVAE(vae_architecture, datasets.Datasets(config["data"]["path"]))
        train_vae.train()

    def test_utils(self):
        """
        Teste les fonctions utilitaires
        """
        utils_loader = utils.Utils()
        self.assertIsNotNone(utils_loader.load_json(config["config"]["path"]))
        self.assertIsNotNone(utils_loader.save_json(config["config"]["path"]))

if __name__ == "__main__":
    unittest.main()
```

```python
# src/config.py

"""
Fichier de configuration du modèle
"""

class Config:
    def __init__(self):
        with open("src/config.json", "r") as file:
            self.config = json.load(file)

    def get_config(self):
        return self.config

config = Config()