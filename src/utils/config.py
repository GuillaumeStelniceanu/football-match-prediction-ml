"""
Configuration du projet
"""

import os
import json
from pathlib import Path

class Config:
    """Classe de configuration du projet."""
    
    # Chemins par défaut
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = BASE_DIR / 'data'
    MODELS_DIR = BASE_DIR / 'models'
    VISUALS_DIR = BASE_DIR / 'visuals'
    SCRIPTS_DIR = BASE_DIR / 'scripts'
    
    def __init__(self, config_file=None):
        """
        Initialise la configuration.
        
        Parameters:
        -----------
        config_file : str, optional
            Chemin vers le fichier de configuration
        """
        self.config = self.load_default_config()
        
        if config_file and os.path.exists(config_file):
            self.load_config_file(config_file)
        
        # Créer les répertoires s'ils n'existent pas
        self.create_directories()
    
    def load_default_config(self):
        """
        Charge la configuration par défaut.
        
        Returns:
        --------
        dict
            Configuration par défaut
        """
        return {
            'paths': {
                'raw_data': str(self.DATA_DIR / 'raw' / 'matches.csv'),
                'processed_data': str(self.DATA_DIR / 'processed' / 'matches_processed.csv'),
                'features_data': str(self.DATA_DIR / 'processed' / 'matches_with_features.csv'),
                'models_dir': str(self.MODELS_DIR),
                'visuals_dir': str(self.VISUALS_DIR)
            },
            'preprocessing': {
                'test_size': 0.2,
                'random_state': 42,
                'target_column': 'ResultCode'
            },
            'feature_engineering': {
                'window_size': 5,
                'k_features': 15
            },
            'model_training': {
                'models': ['Logistic Regression', 'Random Forest', 'XGBoost'],
                'test_size': 0.2,
                'random_state': 42
            },
            'evaluation': {
                'cv_folds': 5,
                'metrics': ['accuracy', 'f1_score', 'precision', 'recall']
            }
        }
    
    def load_config_file(self, config_file):
        """
        Charge la configuration depuis un fichier.
        
        Parameters:
        -----------
        config_file : str
            Chemin vers le fichier de configuration
        """
        with open(config_file, 'r') as f:
            user_config = json.load(f)
        
        # Fusionner avec la configuration par défaut
        self.merge_configs(self.config, user_config)
    
    def merge_configs(self, default, user):
        """
        Fusionne deux configurations.
        
        Parameters:
        -----------
        default : dict
            Configuration par défaut
        user : dict
            Configuration utilisateur
        """
        for key, value in user.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self.merge_configs(default[key], value)
            else:
                default[key] = value
    
    def create_directories(self):
        """Crée les répertoires nécessaires."""
        directories = [
            self.DATA_DIR / 'raw',
            self.DATA_DIR / 'processed',
            self.MODELS_DIR,
            self.VISUALS_DIR,
            self.SCRIPTS_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_path(self, key):
        """
        Récupère un chemin de configuration.
        
        Parameters:
        -----------
        key : str
            Clé du chemin
            
        Returns:
        --------
        str
            Chemin configuré
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if k in value:
                value = value[k]
            else:
                raise KeyError(f"Cle de configuration non trouvee : {key}")
        
        return value
    
    def save_config(self, save_path):
        """
        Sauvegarde la configuration.
        
        Parameters:
        -----------
        save_path : str
            Chemin de sauvegarde
        """
        with open(save_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def __getitem__(self, key):
        """Opérateur [] pour accéder à la configuration."""
        return self.get_path(key)
    
    def __str__(self):
        """Représentation textuelle de la configuration."""
        return json.dumps(self.config, indent=2)