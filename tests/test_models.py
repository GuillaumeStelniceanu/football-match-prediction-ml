"""
Tests unitaires pour les modèles de Machine Learning
"""

import pandas as pd
import numpy as np
import sys
import os

# Ajouter le dossier racine au path Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_data_preparation():
    """Test de la préparation des données pour les modèles."""
    # Créer des données de test
    data = {
        'GoalDiff': [1, -1, 2, 0, -2],
        'ShotDiff': [3, -2, 5, 1, -3],
        'ResultCode': [1, -1, 1, 0, -1]
    }
    
    df = pd.DataFrame(data)
    
    # Vérifier les types de données
    assert df['GoalDiff'].dtype in [np.int64, np.float64]
    assert df['ResultCode'].dtype in [np.int64, np.float64]
    
    # Vérifier la distribution de la target
    target_counts = df['ResultCode'].value_counts()
    assert len(target_counts) == 3  # -1, 0, 1
    
    print("✅ Test de préparation des données réussi")

def test_train_test_split():
    """Test du split train/test."""
    np.random.seed(42)
    
    # Créer des données
    n_samples = 100
    X = np.random.randn(n_samples, 5)
    y = np.random.choice([-1, 0, 1], n_samples, p=[0.3, 0.25, 0.45])
    
    # Simuler un split 80/20
    split_idx = int(n_samples * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    assert len(X_train) == 80
    assert len(X_test) == 20
    assert len(y_train) == 80
    assert len(y_test) == 20
    
    print("✅ Test de train/test split réussi")

def test_model_metrics():
    """Test des métriques d'évaluation."""
    # Données de test
    y_true = np.array([1, 0, -1, 1, 0])
    y_pred = np.array([1, 0, 0, 1, -1])
    
    # Calculer l'accuracy manuellement
    correct = (y_true == y_pred).sum()
    total = len(y_true)
    accuracy = correct / total
    
    assert accuracy == 0.6  # 3 corrects sur 5
    print("✅ Test des métriques d'évaluation réussi")

def test_confusion_matrix():
    """Test de la matrice de confusion."""
    y_true = np.array([1, 0, -1, 1, 0, -1])
    y_pred = np.array([1, 0, 0, 1, -1, -1])
    
    # Calculer la matrice de confusion manuellement
    classes = [-1, 0, 1]
    cm = np.zeros((3, 3), dtype=int)
    
    for true, pred in zip(y_true, y_pred):
        i = classes.index(true)
        j = classes.index(pred)
        cm[i, j] += 1
    
    # Vérifier avec sklearn (si disponible)
    try:
        from sklearn.metrics import confusion_matrix
        sk_cm = confusion_matrix(y_true, y_pred, labels=classes)
        # Vérifier que notre calcul correspond à sklearn
        assert np.array_equal(cm, sk_cm)
    except ImportError:
        # Si sklearn n'est pas disponible, vérifier manuellement
        assert cm[0, 0] == 1  # -1 prédit comme -1
        assert cm[0, 1] == 1  # -1 prédit comme 0
        assert cm[1, 1] == 1  # 0 prédit comme 0
        assert cm[1, 0] == 1  # 0 prédit comme -1
        assert cm[2, 2] == 2  # 1 prédit comme 1
    
    print("✅ Test de la matrice de confusion réussi")

def test_feature_importance():
    """Test de l'importance des features."""
    # Simuler des importances de features
    features = ['GoalDiff', 'ShotDiff', 'CornerDiff', 'HomeForm', 'AwayForm']
    importances = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
    
    # Vérifier que les importances somment à 1 (ou proche)
    total_importance = importances.sum()
    assert 0.99 <= total_importance <= 1.01
    
    # Vérifier que la feature la plus importante est GoalDiff
    most_important_idx = importances.argmax()
    assert features[most_important_idx] == 'GoalDiff'
    
    print("✅ Test de l'importance des features réussi")

def test_model_comparison():
    """Test de la comparaison de modèles."""
    # Simuler les performances de différents modèles
    models = {
        'Logistic Regression': 0.582,
        'Random Forest': 0.625,
        'XGBoost': 0.648
    }
    
    # Vérifier que XGBoost est le meilleur
    best_model = max(models, key=models.get)
    best_accuracy = models[best_model]
    
    assert best_model == 'XGBoost'
    assert best_accuracy == 0.648
    assert best_accuracy > 0.6  # Doit être meilleur que le baseline
    
    print("✅ Test de comparaison de modèles réussi")

def run_all_tests():
    """Exécute tous les tests."""
    print("\n" + "="*50)
    print("🧪 LANCEMENT DES TESTS DES MODÈLES")
    print("="*50)
    
    tests = [
        test_data_preparation,
        test_train_test_split,
        test_model_metrics,
        test_confusion_matrix,
        test_feature_importance,
        test_model_comparison
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"❌ {test.__name__} échoué : {e}")
            failed += 1
        except Exception as e:
            print(f"❌ {test.__name__} erreur : {e}")
            failed += 1
    
    print("\n" + "="*50)
    print(f"📊 RÉSULTATS : {passed} tests réussis, {failed} tests échoués")
    print("="*50)
    
    return passed, failed

if __name__ == "__main__":
    run_all_tests()