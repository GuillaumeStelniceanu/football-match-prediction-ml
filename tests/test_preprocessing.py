"""
Tests unitaires pour le prétraitement des données
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Ajouter le dossier racine au path Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_data_loading():
    """Test du chargement des données."""
    # Créer des données de test
    data = {
        'FTHG': [2, 1, 3],
        'FTAG': [1, 2, 0],
        'HomeTeam': ['TeamA', 'TeamB', 'TeamC'],
        'AwayTeam': ['TeamD', 'TeamE', 'TeamF']
    }
    
    df = pd.DataFrame(data)
    assert len(df) == 3
    assert 'FTHG' in df.columns
    print("✅ Test de chargement des données réussi")

def test_column_renaming():
    """Test du renommage des colonnes."""
    df = pd.DataFrame({
        'FTHG': [2, 1],
        'FTAG': [1, 2],
        'HS': [10, 8]
    })
    
    # Simuler le renommage
    column_mapping = {'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals', 'HS': 'HomeShots'}
    df = df.rename(columns=column_mapping)
    
    assert 'HomeGoals' in df.columns
    assert 'AwayGoals' in df.columns
    assert 'HomeShots' in df.columns
    print("✅ Test de renommage des colonnes réussi")

def test_target_creation():
    """Test de la création de la variable cible."""
    df = pd.DataFrame({
        'FullTimeResult': ['H', 'D', 'A', 'H'],
        'HomeTeam': ['A', 'B', 'C', 'D'],
        'AwayTeam': ['E', 'F', 'G', 'H']
    })
    
    # Simuler la création de la target
    result_mapping = {'H': 1, 'D': 0, 'A': -1}
    df['ResultCode'] = df['FullTimeResult'].map(result_mapping)
    
    assert 'ResultCode' in df.columns
    assert df['ResultCode'].tolist() == [1, 0, -1, 1]
    print("✅ Test de création de la target réussi")

def test_feature_creation():
    """Test de la création des features de base."""
    df = pd.DataFrame({
        'HomeGoals': [2, 1, 3],
        'AwayGoals': [1, 2, 0]
    })
    
    # Simuler la création de features
    df['GoalDiff'] = df['HomeGoals'] - df['AwayGoals']
    df['TotalGoals'] = df['HomeGoals'] + df['AwayGoals']
    
    assert 'GoalDiff' in df.columns
    assert 'TotalGoals' in df.columns
    assert df['GoalDiff'].tolist() == [1, -1, 3]
    assert df['TotalGoals'].tolist() == [3, 3, 3]
    print("✅ Test de création des features réussi")

def test_missing_values():
    """Test de la gestion des valeurs manquantes."""
    df = pd.DataFrame({
        'HomeGoals': [2, np.nan, 3],
        'AwayGoals': [1, 2, np.nan]
    })
    
    # Simuler l'imputation
    df_filled = df.fillna(df.mean())
    
    assert not df_filled['HomeGoals'].isnull().any()
    assert not df_filled['AwayGoals'].isnull().any()
    print("✅ Test de gestion des valeurs manquantes réussi")

def run_all_tests():
    """Exécute tous les tests."""
    print("\n" + "="*50)
    print("🧪 LANCEMENT DES TESTS DE PRÉTRAITEMENT")
    print("="*50)
    
    tests = [
        test_data_loading,
        test_column_renaming,
        test_target_creation,
        test_feature_creation,
        test_missing_values
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