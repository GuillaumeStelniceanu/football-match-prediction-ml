"""
Entraînement des modèles de prédiction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import joblib
import os
import json
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    """Classe pour entraîner les modèles de prédiction."""
    
    def __init__(self, test_size=0.2, random_state=42):
        """
        Initialise l'entraîneur de modèles.
        
        Parameters:
        -----------
        test_size : float
            Proportion des données de test
        random_state : int
            Seed pour la reproductibilité
        """
        self.test_size = test_size
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.features = []
    
    def prepare_data(self, df, target_col='ResultCode'):
        """
        Prépare les données pour l'entraînement.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame d'entrée
        target_col : str
            Colonne cible
            
        Returns:
        --------
        tuple
            Données préparées (X_train, X_test, y_train, y_test)
        """
        print("Preparation des donnees...")
        
        # Sélectionner les features numériques
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Enlever la target et les colonnes encodées pour XGBoost si présent
        features = [col for col in numeric_cols if col != target_col]
        
        # Garder un nombre raisonnable de features
        self.features = features[:20] if len(features) > 20 else features
        
        X = df[self.features]
        y = df[target_col]
        
        print(f"Features utilisees : {len(self.features)}")
        print(f"Target classes : {sorted(y.unique().tolist())}")
        
        # Split temporel si dates disponibles
        if 'Date' in df.columns:
            print("Split temporel des donnees...")
            df_sorted = df.sort_values('Date')
            X_sorted = X.loc[df_sorted.index]
            y_sorted = y.loc[df_sorted.index]
            
            split_idx = int(len(X_sorted) * (1 - self.test_size))
            X_train, X_test = X_sorted.iloc[:split_idx], X_sorted.iloc[split_idx:]
            y_train, y_test = y_sorted.iloc[:split_idx], y_sorted.iloc[split_idx:]
        else:
            # Split aléatoire stratifié
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
            )
        
        print(f"Train size : {len(X_train)}")
        print(f"Test size  : {len(X_test)}")
        
        # Normalisation
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_logistic_regression(self, X_train, y_train):
        """
        Entraîne un modèle de régression logistique.
        
        Parameters:
        -----------
        X_train : array-like
            Features d'entraînement
        y_train : array-like
            Target d'entraînement
            
        Returns:
        --------
        LogisticRegression
            Modèle entraîné
        """
        print("\nEntrainement Logistic Regression...")
        model = LogisticRegression(
            max_iter=1000,
            random_state=self.random_state,
            solver='lbfgs'
        )
        model.fit(X_train, y_train)
        self.models['Logistic Regression'] = model
        return model
    
    def train_random_forest(self, X_train, y_train):
        """
        Entraîne un modèle Random Forest.
        
        Parameters:
        -----------
        X_train : array-like
            Features d'entraînement
        y_train : array-like
            Target d'entraînement
            
        Returns:
        --------
        RandomForestClassifier
            Modèle entraîné
        """
        print("\nEntrainement Random Forest...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=self.random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        self.models['Random Forest'] = model
        return model
    
    def train_xgboost(self, X_train, y_train):
        """
        Entraîne un modèle XGBoost.
        
        Parameters:
        -----------
        X_train : array-like
            Features d'entraînement
        y_train : array-like
            Target d'entraînement
            
        Returns:
        --------
        XGBClassifier
            Modèle entraîné
        """
        print("\nEntrainement XGBoost...")
        
        # XGBoost nécessite des labels [0, 1, 2]
        y_train_encoded = y_train.map({-1: 0, 0: 1, 1: 2})
        
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        model.fit(X_train, y_train_encoded)
        self.models['XGBoost'] = model
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """
        Évalue un modèle.
        
        Parameters:
        -----------
        model : object
            Modèle à évaluer
        X_test : array-like
            Features de test
        y_test : array-like
            Target de test
        model_name : str
            Nom du modèle
            
        Returns:
        --------
        dict
            Résultats d'évaluation
        """
        # Prédictions
        if model_name == 'XGBoost':
            y_pred_encoded = model.predict(X_test)
            y_pred = np.array([-1 if x == 0 else (0 if x == 1 else 1) for x in y_pred_encoded])
        else:
            y_pred = model.predict(X_test)
        
        # Métriques
        accuracy = accuracy_score(y_test, y_pred)
        
        # Rapport de classification
        report = classification_report(y_test, y_pred, output_dict=True)
        
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'predictions': y_pred.tolist()
        }
        
        self.results[model_name] = results
        
        print(f"  Accuracy : {accuracy:.3f}")
        print(f"  Rapport :")
        print(f"    - Precision (macro) : {report['macro avg']['precision']:.3f}")
        print(f"    - Recall (macro)    : {report['macro avg']['recall']:.3f}")
        print(f"    - F1-Score (macro)  : {report['macro avg']['f1-score']:.3f}")
        
        return results
    
    def train_all_models(self, df, target_col='ResultCode'):
        """
        Entraîne tous les modèles.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame d'entrée
        target_col : str
            Colonne cible
            
        Returns:
        --------
        dict
            Modèles entraînés
        """
        print("\n" + "="*50)
        print("DEBUT DE L'ENTRAINEMENT")
        print("="*50)
        
        # Préparation des données
        X_train, X_test, y_train, y_test = self.prepare_data(df, target_col)
        
        # Entraînement des modèles
        self.train_logistic_regression(X_train, y_train)
        self.train_random_forest(X_train, y_train)
        self.train_xgboost(X_train, y_train)
        
        # Évaluation
        print("\n" + "="*50)
        print("EVALUATION DES MODELES")
        print("="*50)
        
        for model_name, model in self.models.items():
            print(f"\nEvaluation {model_name}:")
            self.evaluate_model(model, X_test, y_test, model_name)
        
        # Trouver le meilleur modèle
        self.best_model_name = max(self.results, key=lambda x: self.results[x]['accuracy'])
        self.best_model = self.models[self.best_model_name]
        
        print(f"\nMEILLEUR MODELE : {self.best_model_name}")
        print(f"Accuracy : {self.results[self.best_model_name]['accuracy']:.3f}")
        
        return self.models
    
    def save_models(self, save_dir='../models'):
        """
        Sauvegarde les modèles entraînés.
        
        Parameters:
        -----------
        save_dir : str
            Répertoire de sauvegarde
        """
        print(f"\nSauvegarde des modeles dans {save_dir}...")
        os.makedirs(save_dir, exist_ok=True)
        
        # Sauvegarder chaque modèle
        for model_name, model in self.models.items():
            filename = model_name.lower().replace(' ', '_') + '.pkl'
            filepath = os.path.join(save_dir, filename)
            joblib.dump(model, filepath)
            print(f"  {model_name:20} -> {filepath}")
        
        # Sauvegarder le scaler
        scaler_path = os.path.join(save_dir, 'scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        print(f"  Scaler              -> {scaler_path}")
        
        # Sauvegarder la liste des features
        features_path = os.path.join(save_dir, 'features_used.txt')
        with open(features_path, 'w') as f:
            for feature in self.features:
                f.write(f"{feature}\n")
        print(f"  Features            -> {features_path}")
        
        # Sauvegarder les résultats
        results_path = os.path.join(save_dir, 'training_results.json')
        with open(results_path, 'w') as f:
            # Convertir les résultats en format JSON serializable
            json_results = {}
            for model_name, results in self.results.items():
                json_results[model_name] = {
                    'accuracy': float(results['accuracy']),
                    'predictions': results['predictions']
                }
            json.dump(json_results, f, indent=2)
        print(f"  Resultats           -> {results_path}")
        
        print("\nSauvegarde terminee.")
    
    def get_training_summary(self):
        """
        Génère un résumé de l'entraînement.
        
        Returns:
        --------
        dict
            Résumé de l'entraînement
        """
        summary = {
            'models_trained': list(self.models.keys()),
            'best_model': self.best_model_name if hasattr(self, 'best_model_name') else None,
            'features_used': len(self.features),
            'results': {}
        }
        
        for model_name, results in self.results.items():
            summary['results'][model_name] = {
                'accuracy': float(results['accuracy'])
            }
        
        return summary