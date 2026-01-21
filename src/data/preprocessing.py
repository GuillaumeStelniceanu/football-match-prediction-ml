"""
Prétraitement des données de matchs de football
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Classe pour le prétraitement des données de matchs."""
    
    def __init__(self, data_path=None):
        """
        Initialise le prétraitement.
        
        Parameters:
        -----------
        data_path : str, optional
            Chemin vers le fichier de données
        """
        self.data_path = data_path
        self.df = None
        self.features = None
        
    def load_data(self, data_path=None):
        """
        Charge les données depuis un fichier CSV.
        
        Parameters:
        -----------
        data_path : str, optional
            Chemin vers le fichier de données
            
        Returns:
        --------
        pandas.DataFrame
            Données chargées
        """
        if data_path is None:
            data_path = self.data_path
        
        if data_path is None:
            raise ValueError("Le chemin des données doit être spécifié")
        
        try:
            self.df = pd.read_csv(data_path)
            print(f"Donnees chargees : {self.df.shape[0]} matchs, {self.df.shape[1]} colonnes")
            return self.df
        except FileNotFoundError:
            print(f"Fichier non trouve : {data_path}")
            raise
    
    def rename_columns(self, df):
        """
        Renomme les colonnes pour plus de clarté.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame d'entrée
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame avec colonnes renommées
        """
        column_mapping = {
            'Date': 'Date',
            'HomeTeam': 'HomeTeam',
            'AwayTeam': 'AwayTeam',
            'FTHG': 'HomeGoals',
            'FTAG': 'AwayGoals',
            'FTR': 'FullTimeResult',
            'HTHG': 'HT_HomeGoals',
            'HTAG': 'HT_AwayGoals',
            'HTR': 'HalfTimeResult',
            'Referee': 'Referee',
            'HS': 'HomeShots',
            'AS': 'AwayShots',
            'HST': 'HomeShotsTarget',
            'AST': 'AwayShotsTarget',
            'HF': 'HomeFouls',
            'AF': 'AwayFouls',
            'HC': 'HomeCorners',
            'AC': 'AwayCorners',
            'HY': 'HomeYellowCards',
            'AY': 'AwayYellowCards',
            'HR': 'HomeRedCards',
            'AR': 'AwayRedCards'
        }
        
        # Garder seulement les colonnes qui existent
        existing_columns = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=existing_columns)
        
        return df
    
    def create_target(self, df):
        """
        Crée la variable cible à partir du résultat.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame d'entrée
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame avec la target ajoutée
        """
        if 'FullTimeResult' in df.columns:
            result_mapping = {'H': 1, 'D': 0, 'A': -1}
            df['ResultCode'] = df['FullTimeResult'].map(result_mapping)
            print("Target ResultCode creee")
        else:
            print("Attention : colonne FullTimeResult non trouvee")
        
        return df
    
    def create_basic_features(self, df):
        """
        Crée les features de base (différences).
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame d'entrée
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame avec features de base
        """
        # Différences de base
        if 'HomeGoals' in df.columns and 'AwayGoals' in df.columns:
            df['GoalDiff'] = df['HomeGoals'] - df['AwayGoals']
            df['TotalGoals'] = df['HomeGoals'] + df['AwayGoals']
        
        if 'HomeShots' in df.columns and 'AwayShots' in df.columns:
            df['ShotDiff'] = df['HomeShots'] - df['AwayShots']
        
        if 'HomeShotsTarget' in df.columns and 'AwayShotsTarget' in df.columns:
            df['ShotOnTargetDiff'] = df['HomeShotsTarget'] - df['AwayShotsTarget']
        
        if 'HomeCorners' in df.columns and 'AwayCorners' in df.columns:
            df['CornerDiff'] = df['HomeCorners'] - df['AwayCorners']
        
        if 'HomeFouls' in df.columns and 'AwayFouls' in df.columns:
            df['FoulDiff'] = df['HomeFouls'] - df['AwayFouls']
        
        if 'HomeYellowCards' in df.columns and 'AwayYellowCards' in df.columns:
            df['YellowCardDiff'] = df['HomeYellowCards'] - df['AwayYellowCards']
        
        return df
    
    def handle_missing_values(self, df, strategy='mean'):
        """
        Gère les valeurs manquantes.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame d'entrée
        strategy : str
            Stratégie d'imputation ('mean', 'median', 'mode')
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame sans valeurs manquantes
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Pour les colonnes numériques
        if strategy == 'mean':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif strategy == 'median':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # Pour les colonnes catégorielles
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
        print(f"Valeurs manquantes traitees (strategie: {strategy})")
        return df
    
    def convert_dates(self, df):
        """
        Convertit les dates au format datetime.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame d'entrée
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame avec dates converties
        """
        if 'Date' in df.columns:
            try:
                df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%y', errors='coerce')
                print("Dates converties au format datetime")
            except:
                try:
                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                    print("Dates converties (format auto-detecte)")
                except:
                    print("Impossible de convertir les dates")
        
        return df
    
    def preprocess(self, df=None, save_path=None):
        """
        Exécute le pipeline complet de prétraitement.
        
        Parameters:
        -----------
        df : pandas.DataFrame, optional
            DataFrame d'entrée
        save_path : str, optional
            Chemin pour sauvegarder les données traitées
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame prétraité
        """
        if df is None:
            if self.df is None:
                raise ValueError("Aucune donnee disponible. Chargez d'abord les donnees.")
            df = self.df
        
        print("\n" + "="*50)
        print("DEBUT DU PRETRAITEMENT")
        print("="*50)
        
        # 1. Renommage des colonnes
        print("1. Renommage des colonnes...")
        df = self.rename_columns(df)
        
        # 2. Création de la target
        print("2. Creation de la target...")
        df = self.create_target(df)
        
        # 3. Conversion des dates
        print("3. Conversion des dates...")
        df = self.convert_dates(df)
        
        # 4. Création des features de base
        print("4. Creation des features de base...")
        df = self.create_basic_features(df)
        
        # 5. Gestion des valeurs manquantes
        print("5. Gestion des valeurs manquantes...")
        df = self.handle_missing_values(df)
        
        # Sauvegarde si demandé
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df.to_csv(save_path, index=False)
            print(f"\nDonnees pretraitees sauvegardees : {save_path}")
        
        print("\n" + "="*50)
        print("PRETRAITEMENT TERMINE")
        print("="*50)
        print(f"Shape final : {df.shape}")
        print(f"Colonnes : {len(df.columns)}")
        
        self.df = df
        return df
    
    def get_summary(self):
        """
        Retourne un résumé des données.
        
        Returns:
        --------
        dict
            Résumé des données
        """
        if self.df is None:
            return {}
        
        summary = {
            'n_samples': len(self.df),
            'n_features': len(self.df.columns),
            'target_present': 'ResultCode' in self.df.columns,
            'missing_values': self.df.isnull().sum().sum(),
            'date_range': None
        }
        
        if 'Date' in self.df.columns:
            summary['date_range'] = {
                'min': self.df['Date'].min(),
                'max': self.df['Date'].max()
            }
        
        if 'ResultCode' in self.df.columns:
            summary['target_distribution'] = self.df['ResultCode'].value_counts().to_dict()
        
        return summary