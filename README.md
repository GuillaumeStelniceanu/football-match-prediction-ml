# ⚽ Football Match Prediction - Machine Learning

## 📊 Description
Système de prédiction de matchs de football utilisant le Machine Learning. Analyse des 5 grands championnats européens (2022-2023) pour prédire Victoire Domicile/Nul/Victoire Extérieur.

## 🚀 Installation rapide

```bash
# 1. Cloner le projet
git clone https://github.com/username/football-match-prediction-ml.git
cd football-match-prediction-ml

# 2. Créer l'environnement virtuel
python -m venv venv

# Sur Windows
venv\Scripts\activate

# Sur Mac/Linux
source venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt
```
football-match-prediction-ml/
├── data/              # Données
├── notebooks/         # Analyses Jupyter
├── src/              # Code source
├── models/           # Modèles sauvegardés
├── visuals/          # Graphiques
└── requirements.txt  # Dépendances

## 🎯 Utilisation
# - Pipeline complet :
    python main.py --mode full
# - Étapes individuelles :
    python main.py --mode preprocess

    python main.py --mode train --model xgboost

    python main.py --mode predict --input nouveau_match.csv

    python main.py --mode visualize

## 📈 Modèles implémentés
- Régression Logistique - Baseline
- Random Forest - Modèle ensembliste
- XGBoost - Meilleures performances (64.8%)
- Réseau de Neurones - Approche deep learning

## 🔧 Technologies
- Python 3.9+
- Scikit-learn, XGBoost, TensorFlow
- Pandas, NumPy
- Matplotlib, Seaborn

## 📞 Contact
STELNICEANU Guillaume - g.stelniceanu@gmail.com 
Projet GitHub: football-match-prediction-ml

