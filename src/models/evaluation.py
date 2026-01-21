"""
Évaluation avancée des modèles
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, 
                           confusion_matrix, classification_report,
                           precision_score, recall_score)
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """Classe pour l'évaluation avancée des modèles."""
    
    def __init__(self):
        """Initialise l'évaluateur de modèles."""
        self.models = {}
        self.results = {}
        self.cv_results = {}
    
    def load_models(self, models_dir='../models'):
        """
        Charge les modèles depuis un répertoire.
        
        Parameters:
        -----------
        models_dir : str
            Répertoire des modèles
        """
        print("Chargement des modeles...")
        
        model_files = {
            'Logistic Regression': 'logistic_regression.pkl',
            'Random Forest': 'random_forest.pkl',
            'XGBoost': 'xgboost.pkl'
        }
        
        for name, filename in model_files.items():
            filepath = os.path.join(models_dir, filename)
            if os.path.exists(filepath):
                self.models[name] = joblib.load(filepath)
                print(f"  {name:20} charge")
            else:
                print(f"  {name:20} non trouve")
        
        if not self.models:
            raise ValueError("Aucun modele trouve dans le repertoire")
    
    def load_features(self, models_dir='../models'):
        """
        Charge la liste des features utilisées.
        
        Parameters:
        -----------
        models_dir : str
            Répertoire des modèles
            
        Returns:
        --------
        list
            Liste des features
        """
        features_path = os.path.join(models_dir, 'features_used.txt')
        if os.path.exists(features_path):
            with open(features_path, 'r') as f:
                features = [line.strip() for line in f]
            print(f"Features chargees : {len(features)}")
            return features
        else:
            print("Fichier de features non trouve")
            return []
    
    def prepare_evaluation_data(self, df, features):
        """
        Prépare les données pour l'évaluation.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame d'entrée
        features : list
            Liste des features à utiliser
            
        Returns:
        --------
        tuple
            Données préparées (X, y)
        """
        # Vérifier que toutes les features sont présentes
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            print(f"Features manquantes : {missing_features}")
            # Utiliser seulement les features disponibles
            features = [f for f in features if f in df.columns]
        
        X = df[features]
        
        # Target
        if 'ResultCode' in df.columns:
            y = df['ResultCode']
        elif 'ResultCode_encoded' in df.columns:
            y_encoded = df['ResultCode_encoded']
            y = y_encoded.map({0: -1, 1: 0, 2: 1})
        else:
            raise ValueError("Target non trouvee dans les donnees")
        
        print(f"Donnees preparees : X={X.shape}, y={y.shape}")
        return X, y
    
    def cross_validate_models(self, X, y, cv_splits=5):
        """
        Effectue une cross-validation temporelle.
        
        Parameters:
        -----------
        X : array-like
            Features
        y : array-like
            Target
        cv_splits : int
            Nombre de folds
            
        Returns:
        --------
        dict
            Résultats de cross-validation
        """
        print(f"\nCross-validation ({cv_splits} folds)...")
        
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        self.cv_results = {}
        
        for name, model in self.models.items():
            print(f"\n  {name}:")
            try:
                scores = cross_val_score(model, X, y, 
                                        cv=tscv, scoring='accuracy', n_jobs=-1)
                
                self.cv_results[name] = {
                    'scores': scores.tolist(),
                    'mean': float(scores.mean()),
                    'std': float(scores.std())
                }
                
                print(f"    Scores : {scores.round(3)}")
                print(f"    Moyenne : {scores.mean():.3f} (+/- {scores.std()*2:.3f})")
                
            except Exception as e:
                print(f"    Erreur : {e}")
        
        return self.cv_results
    
    def evaluate_on_test_set(self, X_test, y_test):
        """
        Évalue les modèles sur un ensemble de test.
        
        Parameters:
        -----------
        X_test : array-like
            Features de test
        y_test : array-like
            Target de test
            
        Returns:
        --------
        dict
            Résultats d'évaluation
        """
        print("\nEvaluation sur ensemble de test...")
        
        self.results = {}
        
        for name, model in self.models.items():
            print(f"\n  {name}:")
            
            try:
                # Prédictions
                if name == 'XGBoost':
                    y_pred_encoded = model.predict(X_test)
                    y_pred = np.array([-1 if x == 0 else (0 if x == 1 else 1) for x in y_pred_encoded])
                else:
                    y_pred = model.predict(X_test)
                
                # Métriques
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                # Matrice de confusion
                cm = confusion_matrix(y_test, y_pred)
                
                self.results[name] = {
                    'accuracy': float(accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1),
                    'confusion_matrix': cm.tolist()
                }
                
                print(f"    Accuracy  : {accuracy:.3f}")
                print(f"    F1-Score  : {f1:.3f}")
                
            except Exception as e:
                print(f"    Erreur : {e}")
        
        return self.results
    
    def optimize_random_forest(self, X_train, y_train):
        """
        Optimise les hyperparamètres de Random Forest.
        
        Parameters:
        -----------
        X_train : array-like
            Features d'entraînement
        y_train : array-like
            Target d'entraînement
            
        Returns:
        --------
        GridSearchCV
            Modèle optimisé
        """
        print("\nOptimisation de Random Forest...")
        
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        rf = self.models.get('Random Forest')
        if rf is None:
            from sklearn.ensemble import RandomForestClassifier
            rf = RandomForestClassifier(random_state=42)
        
        grid_search = GridSearchCV(
            rf,
            param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"  Meilleurs parametres : {grid_search.best_params_}")
        print(f"  Best score : {grid_search.best_score_:.3f}")
        
        return grid_search
    
    def analyze_feature_importance(self, features):
        """
        Analyse l'importance des features.
        
        Parameters:
        -----------
        features : list
            Liste des features
            
        Returns:
        --------
        dict
            Importance des features par modèle
        """
        print("\nAnalyse de l'importance des features...")
        
        importance_results = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                
                # Associer les importances aux noms de features
                importance_df = pd.DataFrame({
                    'feature': features[:len(importances)],
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                importance_results[name] = importance_df
                
                print(f"\n  {name} - Top 5 features:")
                for i, row in importance_df.head(5).iterrows():
                    print(f"    {row['feature']:25} : {row['importance']:.3f}")
        
        return importance_results
    
    def generate_report(self, save_dir='../models'):
        """
        Génère un rapport d'évaluation.
        
        Parameters:
        -----------
        save_dir : str
            Répertoire de sauvegarde
            
        Returns:
        --------
        dict
            Rapport complet
        """
        print(f"\nGeneration du rapport dans {save_dir}...")
        os.makedirs(save_dir, exist_ok=True)
        
        report = {
            'cross_validation': self.cv_results,
            'test_evaluation': self.results,
            'best_model': None,
            'summary': {}
        }
        
        # Trouver le meilleur modèle
        if self.results:
            best_model = max(self.results.items(), key=lambda x: x[1]['accuracy'])
            report['best_model'] = {
                'name': best_model[0],
                'accuracy': best_model[1]['accuracy']
            }
        
        # Générer un résumé
        summary = {
            'n_models': len(self.models),
            'models_evaluated': list(self.models.keys()),
            'has_cv_results': len(self.cv_results) > 0,
            'has_test_results': len(self.results) > 0
        }
        report['summary'] = summary
        
        # Sauvegarder le rapport
        report_path = os.path.join(save_dir, 'evaluation_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Rapport sauvegarde : {report_path}")
        
        return report
    
    def create_visualizations(self, features, importance_results, save_dir='../visuals'):
        """
        Crée des visualisations des résultats.
        
        Parameters:
        -----------
        features : list
            Liste des features
        importance_results : dict
            Résultats d'importance des features
        save_dir : str
            Répertoire de sauvegarde
        """
        print(f"\nCreation des visualisations dans {save_dir}...")
        os.makedirs(save_dir, exist_ok=True)
        
        plt.style.use('default')
        
        # 1. Comparaison des modèles
        if self.results:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Accuracy des modèles
            model_names = list(self.results.keys())
            accuracies = [self.results[name]['accuracy'] for name in model_names]
            
            colors = ['#3498db', '#2ecc71', '#e74c3c']
            bars = axes[0].bar(model_names, accuracies, color=colors[:len(model_names)])
            axes[0].set_title('Accuracy des Modeles', fontsize=14, pad=20)
            axes[0].set_ylabel('Accuracy')
            axes[0].set_ylim([0, 1])
            axes[0].grid(True, alpha=0.3, axis='y')
            
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{acc:.3f}', ha='center', va='bottom')
            
            # Importance des features pour Random Forest
            if 'Random Forest' in importance_results:
                importance_df = importance_results['Random Forest'].head(10)
                axes[1].barh(importance_df['feature'], importance_df['importance'], color='#2ecc71')
                axes[1].set_title('Top 10 Features - Random Forest', fontsize=14, pad=20)
                axes[1].set_xlabel('Importance')
                axes[1].invert_yaxis()
                axes[1].grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
            plt.show()
        
        print("Visualisations creees et sauvegardees")