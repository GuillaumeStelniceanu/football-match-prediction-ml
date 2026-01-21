"""
Football Match Prediction - Package principal
"""

__version__ = "1.0.0"
__author__ = "STELNICEANU Guillaume"
__email__ = "g.stelniceanu@gmail.com"

from .data.preprocessing import DataPreprocessor
from .features.engineering import FeatureEngineer
from .models.training import ModelTrainer
from .models.evaluation import ModelEvaluator

__all__ = [
    'DataPreprocessor',
    'FeatureEngineer', 
    'ModelTrainer',
    'ModelEvaluator'
]