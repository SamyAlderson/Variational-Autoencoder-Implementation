"""
Fichier contenant les données d'entraînement et de test.
"""

import os
import json
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class Dataset:
    """
    Classe représentant un ensemble de données.
    """
    
    def __init__(self, data, labels, file_path):
        """
        Constructeur de la classe Dataset.

        Args:
            data (numpy.ndarray): Données de l'ensemble.
            labels (numpy.ndarray): Étiquettes de l'ensemble.
            file_path (str): Chemin d'accès au fichier contenant les données.
        """
        self.data = data
        self.labels = labels
        self.file_path = file_path

    def load_data(self):
        """
        Charge les données à partir du fichier spécifié.
        """
        try:
            data = pd.read_csv(self.file_path)
            self.data = data.iloc[:, :-1].values
            self.labels = data.iloc[:, -1].values
            return self.data, self.labels
        except FileNotFoundError as e:
            logger.error(f"Le fichier {self.file_path} n'existe pas.")
            raise e
        except pd.errors.EmptyDataError as e:
            logger.error(f"Le fichier {self.file_path} est vide.")
            raise e
        except pd.errors.ParserError as e:
            logger.error(f"Erreur de parsing du fichier {self.file_path}.")
            raise e

class Datasets:
    """
    Classe contenant les ensembles de données d'entraînement et de test.
    """
    
    def __init__(self, train_file_path, test_file_path):
        """
        Constructeur de la classe Datasets.

        Args:
            train_file_path (str): Chemin d'accès au fichier contenant les données d'entraînement.
            test_file_path (str): Chemin d'accès au fichier contenant les données de test.
        """
        self.train_data = None
        self.train_labels = None
        self.test_data = None
        self.test_labels = None
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path

    def load_train_data(self):
        """
        Charge les données d'entraînement à partir du fichier spécifié.
        """
        self.train_data, self.train_labels = Dataset(None, None, self.train_file_path).load_data()

    def load_test_data(self):
        """
        Charge les données de test à partir du fichier spécifié.
        """
        self.test_data, self.test_labels = Dataset(None, None, self.test_file_path).load_data()

def load_datasets(config):
    """
    Charge les données d'entraînement et de test à partir des chemins spécifiés dans la configuration.

    Args:
        config (dict): Dictionnaire de configuration contenant les chemins des fichiers.

    Returns:
        Datasets: Instance de la classe Datasets contenant les ensembles de données.
    """
    datasets = Datasets(config['train_data_file_path'], config['test_data_file_path'])
    datasets.load_train_data()
    datasets.load_test_data()
    return datasets
```

```json
{
    "train_data_file_path": "path/to/train/data.csv",
    "test_data_file_path": "path/to/test/data.csv"
}
```
