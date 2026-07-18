"""
Fichier contenant des fonctions utilitaires pour le projet Variational-Autoencoder-Implementation.
"""

import json
import os
import numpy as np
from typing import Dict, List

def charger_configuration(fichier_configuration: str) -> Dict:
    """
    Charge la configuration du modèle à partir d'un fichier JSON.

    Args:
        fichier_configuration (str): Chemin du fichier de configuration.

    Returns:
        Dict: Dictionnaire contenant la configuration du modèle.

    Raises:
        FileNotFoundError: Si le fichier de configuration n'existe pas.
        json.JSONDecodeError: Si le fichier de configuration n'est pas valide.
    """
    if not os.path.exists(fichier_configuration):
        raise FileNotFoundError(f"Fichier de configuration '{fichier_configuration}' introuvable.")
    with open(fichier_configuration, 'r') as fichier:
        try:
            configuration = json.load(fichier)
            return configuration
        except json.JSONDecodeError as erreur:
            raise json.JSONDecodeError(f"Erreur de décodage du fichier de configuration : {erreur}")

def charger_donnees(fichier_donnees: str) -> List:
    """
    Charge les données d'entraînement à partir d'un fichier CSV.

    Args:
        fichier_donnees (str): Chemin du fichier de données.

    Returns:
        List: Liste contenant les données d'entraînement.

    Raises:
        FileNotFoundError: Si le fichier de données n'existe pas.
    """
    if not os.path.exists(fichier_donnees):
        raise FileNotFoundError(f"Fichier de données '{fichier_donnees}' introuvable.")
    donnees = np.loadtxt(fichier_donnees, delimiter=',')
    return donnees

def enregistrer_resultats(fichier_resultats: str, resultats: Dict) -> None:
    """
    Enregistre les résultats du modèle à partir d'un dictionnaire.

    Args:
        fichier_resultats (str): Chemin du fichier de résultats.
        resultats (Dict): Dictionnaire contenant les résultats du modèle.
    """
    with open(fichier_resultats, 'w') as fichier:
        json.dump(resultats, fichier, indent=4)