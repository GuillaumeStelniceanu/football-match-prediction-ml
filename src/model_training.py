"""
Entraînement des modèles
"""

import joblib
import os

def train_models(X_train, y_train):
    """Entraîne plusieurs modèles de ML"""
    print("\n🤖 ENTRAÎNEMENT DES MODÈLES")
    print("-" * 40)
    
    models = {}
    
    try:
        # 1. Régression Logistique
        from sklearn.linear_model import LogisticRegression
        print("🔧 Entraînement Régression Logistique...")
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)
        models['LogisticRegression'] = lr
        print("   ✅ Terminé")
        
        # 2. Random Forest
        from sklearn.ensemble import RandomForestClassifier
        print("🔧 Entraînement Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        models['RandomForest'] = rf
        print("   ✅ Terminé")
        
        # 3. Gradient Boosting
        from sklearn.ensemble import GradientBoostingClassifier
        print("🔧 Entraînement Gradient Boosting...")
        gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
        gb.fit(X_train, y_train)
        models['GradientBoosting'] = gb
        print("   ✅ Terminé")
        
    except ImportError as e:
        print(f"⚠️  Erreur d'importation: {e}")
        print("   Installation des bibliothèques nécessaire")
        return None
    
    return models

def save_models(models, feature_names=None):
    """Sauvegarde les modèles"""
    os.makedirs('models', exist_ok=True)
    
    print("\n💾 SAUVEGARDE DES MODÈLES")
    print("-" * 40)
    
    for name, model in models.items():
        filename = f'models/{name.replace(" ", "_").lower()}.pkl'
        joblib.dump(model, filename)
        print(f"   ✅ {name}: {filename}")
    
    # Sauvegarder les noms des features
    if feature_names:
        import json
        with open('models/feature_names.json', 'w') as f:
            json.dump(feature_names, f)
        print(f"   ✅ Features: models/feature_names.json")
    
    return True
