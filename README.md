# ⚽ Football Match Prediction - Machine Learning

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ML](https://img.shields.io/badge/Machine-Learning-orange)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/status-active-success.svg)]()

## 🎯 Overview
Système intelligent de prédiction des résultats de matchs de football utilisant des algorithmes de Machine Learning. Ce projet analyse les 5 grands championnats européens (2022-2023) pour prédire les résultats (Victoire Domicile/Nul/Victoire Extérieur) avec une précision de 64.8%.

## 📊 Features
- **🧠 Multiples modèles ML** : Régression Logistique, Random Forest, XGBoost
- **📈 Feature Engineering** : Création de 15+ indicateurs prédictifs
- **🔍 Analyse avancée** : Importance des features, matrices de confusion
- **🎯 Pipeline automatisé** : De la donnée brute à la prédiction
- **📊 Visualisations** : Graphiques interactifs et insights

## 🏗️ Architecture
 football-match-prediction-ml/
 ├── 📁 data/ # Données brutes et transformées
 │ ├── raw/ # Données originales
 │ └── processed/ # Données prétraitées
 ├── 📁 notebooks/ # Notebooks d'analyse
 │ ├── 01_eda.ipynb # Exploration des données
 │ ├── 02_feature_engineering.ipynb
 │ └── 03_model_training.ipynb
 ├── 📁 src/ # Code source Python
 │ ├── preprocessing.py # Pipeline de prétraitement
 │ ├── models.py # Implémentation des modèles
 │ ├── utils.py # Fonctions utilitaires
 │ └── visualization.py # Génération de graphiques
 ├── 📁 models/ # Modèles entraînés (sauvegardés)
 ├── 📁 tests/ # Tests unitaires
 │ ├── test_preprocessing.py
 │ └── test_models.py
 ├── 📁 visuals/ # Graphiques exportés
 ├── 📄 main.py # Script principal
 ├── 📄 requirements.txt # Dépendances
 └── 📄 README.md # Ce fichier


## 🚀 Quick Start

### 1. Installation
```bash
# Cloner le dépôt
git clone https://github.com/username/football-match-prediction-ml.git
cd football-match-prediction-ml

# Créer l'environnement virtuel
python -m venv venv

# Activer l'environnement
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```
### 2. Exécution du pipeline complet
```bash
python main.py --mode full
```
## 💻 Usage
Modes d'exécution
```bash
# Prétraitement uniquement
python main.py --mode preprocess

# Entraînement d'un modèle spécifique
python main.py --mode train --model xgboost

# Prédiction sur de nouvelles données
python main.py --mode predict --input data/new_matches.csv

# Génération des visualisations
python main.py --mode visualize

# Évaluation des modèles
python main.py --mode evaluate
```
## 🔧 Technologies Stack

# Machine Learning
Scikit-learn : Modèles classiques et pipeline  
XGBoost : Gradient boosting optimisé  
TensorFlow/Keras : Réseaux de neurones  

# Data Processing
Pandas : Manipulation des données  
NumPy : Calculs numériques  
SciPy : Statistiques avancées  

# Visualization
Matplotlib : Graphiques statiques  
Seaborn : Visualisations statistiques  
Plotly : Graphiques interactifs  

## 📊 Features Importantes
Les 5 features les plus prédictives identifiées :  
  
GoalDiff (28%) - Différence de buts moyenne  
ShotDiff (22%) - Différence de tirs  
HomeForm (18%) - Forme de l'équipe à domicile  
AwayForm (15%) - Forme de l'équipe à l'extérieur  
CornerDiff (12%) - Différence de corners  

## 📈 Résultats
Performance des modèles
https://visuals/confusion_matrix.png

Importance des features
https://visuals/feature_importance.png

Prédictions vs Réalité
https://visuals/predictions_vs_reality.png

## 🧪 Tests
``` bash
# Exécuter tous les tests
python -m pytest tests/

# Tests spécifiques
python tests/test_preprocessing.py
python tests/test_models.py
```
## 📝 Dataset
Sources
FootyStats API : Données historiques
FBref : Statistiques avancées
Understat : Métriques xG/xA

# Championnats couverts
- Premier League (Angleterre)
- La Liga (Espagne)
- Serie A (Italie)
- Bundesliga (Allemagne)
- Ligue 1 (France)

## 🔮 Roadmap

- Pipeline de prétraitement
- Implémentation modèles ML
- Système d'évaluation
- Tests unitaires
- API de prédiction en temps réel
- Interface web dashboard
- Intégration données live
- Modèles de deep learning avancés

## 👤 Auteur
STELNICEANU Guillaume