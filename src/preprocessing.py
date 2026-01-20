"""
Prétraitement des données
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess():
    """Charge et prépare les données pour le ML"""
    print("📊 Chargement des données...")
    
    try:
        df = pd.read_csv('data/raw/matches.csv')
        print(f"   ✅ {len(df)} matchs chargés")
    except:
        print("   ❌ Fichier non trouvé. Exécutez d'abord prepare_dataset.py")
        return None
    
    # Créer les features
    features = pd.DataFrame()
    
    # Features de base (vérifier leur existence)
    base_features = {
        'GoalDiff': ('HomeGoals', 'AwayGoals'),
        'ShotDiff': ('HomeShots', 'AwayShots'),
        'CornerDiff': ('HomeCorners', 'AwayCorners'),
        'YellowCardDiff': ('HomeYellowCards', 'AwayYellowCards')
    }
    
    for feat_name, (col1, col2) in base_features.items():
        if col1 in df.columns and col2 in df.columns:
            features[feat_name] = df[col1] - df[col2]
            print(f"   ✅ Feature créée: {feat_name}")
    
    # Vérifier qu'on a au moins GoalDiff
    if 'GoalDiff' not in features.columns and 'HomeGoals' in df.columns and 'AwayGoals' in df.columns:
        features['GoalDiff'] = df['HomeGoals'] - df['AwayGoals']
    
    # Target
    if 'Result' in df.columns:
        y = df['Result']
    elif 'GoalDiff' in features.columns:
        y = features['GoalDiff'].apply(lambda x: 1 if x > 0 else (0 if x == 0 else -1))
        print("   ✅ Target créée à partir de GoalDiff")
    else:
        print("   ❌ Impossible de créer la target")
        return None
    
    X = features
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n✅ Données préparées:")
    print(f"   Features: {X.shape[1]} ({', '.join(X.columns.tolist())})")
    print(f"   Train: {X_train.shape[0]} échantillons")
    print(f"   Test: {X_test.shape[0]} échantillons")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist()
