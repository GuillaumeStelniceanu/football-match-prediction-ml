"""
Feature engineering pour les matchs de football
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
import os
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """Classe pour le feature engineering."""
    
    def __init__(self, window_size=5):
        """
        Initialise l'ingénieur de features.
        
        Parameters:
        -----------
        window_size : int
            Taille de la fenêtre pour les calculs de forme
        """
        self.window_size = window_size
        self.scaler = StandardScaler()
        self.team_encoder = LabelEncoder()
        self.selected_features = []
    
    def calculate_team_form(self, df, team_col, window=None):
        """
        Calcule la forme récente d'une équipe.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame des matchs
        team_col : str
            Nom de la colonne de l'équipe
        window : int, optional
            Taille de la fenêtre
            
        Returns:
        --------
        numpy.ndarray
            Scores de forme
        """
        if window is None:
            window = self.window_size
        
        form_scores = []
        df_sorted = df.sort_values('Date') if 'Date' in df.columns else df
        
        for i, row in df_sorted.iterrows():
            team = row[team_col]
            
            # Trouver les matchs précédents de cette équipe
            if team_col == 'HomeTeam':
                mask = (df_sorted['HomeTeam'] == team) | (df_sorted['AwayTeam'] == team)
            else:
                mask = (df_sorted['HomeTeam'] == team) | (df_sorted['AwayTeam'] == team)
            
            # Matchs précédents (avant le match courant)
            previous_idx = df_sorted.index[:i]
            previous_matches = df_sorted.loc[previous_idx][mask].tail(window)
            
            if len(previous_matches) > 0:
                # Calculer les points (3 pour victoire, 1 pour nul, 0 pour défaite)
                total_points = 0
                for _, match in previous_matches.iterrows():
                    if match['HomeTeam'] == team:
                        if match['ResultCode'] == 1:
                            total_points += 3
                        elif match['ResultCode'] == 0:
                            total_points += 1
                    else:  # AwayTeam
                        if match['ResultCode'] == -1:
                            total_points += 3
                        elif match['ResultCode'] == 0:
                            total_points += 1
                
                form_score = total_points / (len(previous_matches) * 3)  # Normalisé [0, 1]
            else:
                form_score = 0.5  # Valeur par défaut
            
            form_scores.append(form_score)
        
        return np.array(form_scores)
    
    def calculate_average_stats(self, df, team_col, stat_cols, window=None):
        """
        Calcule les statistiques moyennes d'une équipe.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame des matchs
        team_col : str
            Nom de la colonne de l'équipe
        stat_cols : list
            Liste des colonnes de statistiques
        window : int, optional
            Taille de la fenêtre
            
        Returns:
        --------
        dict
            Dictionnaire des statistiques moyennes
        """
        if window is None:
            window = self.window_size
        
        averages = {f'{col}_avg': [] for col in stat_cols}
        df_sorted = df.sort_values('Date') if 'Date' in df.columns else df
        
        for i, row in df_sorted.iterrows():
            team = row[team_col]
            
            # Déterminer si l'équipe est à domicile ou à l'extérieur
            if team_col == 'HomeTeam':
                mask = df_sorted['HomeTeam'] == team
                prefix = 'Home'
            else:
                mask = df_sorted['AwayTeam'] == team
                prefix = 'Away'
            
            # Matchs précédents
            previous_idx = df_sorted.index[:i]
            previous_matches = df_sorted.loc[previous_idx][mask].tail(window)
            
            for col in stat_cols:
                if len(previous_matches) > 0:
                    avg_value = previous_matches[col].mean()
                else:
                    # Valeur moyenne globale par défaut
                    avg_value = df[col].mean() if col in df.columns else 0
                
                averages[f'{col}_avg'].append(avg_value)
        
        return averages
    
    def create_team_features(self, df):
        """
        Crée les features liées aux équipes.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame d'entrée
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame avec features d'équipes
        """
        print("Creation des features d'equipes...")
        
        # Forme récente
        if 'HomeTeam' in df.columns and 'ResultCode' in df.columns:
            df['HomeForm'] = self.calculate_team_form(df, 'HomeTeam')
            df['AwayForm'] = self.calculate_team_form(df, 'AwayTeam')
            df['FormDiff'] = df['HomeForm'] - df['AwayForm']
        
        # Statistiques moyennes
        home_stats = ['HomeGoals', 'HomeShots', 'HomeShotsTarget', 'HomeCorners']
        away_stats = ['AwayGoals', 'AwayShots', 'AwayShotsTarget', 'AwayCorners']
        
        home_stats = [col for col in home_stats if col in df.columns]
        away_stats = [col for col in away_stats if col in df.columns]
        
        if home_stats:
            home_averages = self.calculate_average_stats(df, 'HomeTeam', home_stats)
            for col_name, values in home_averages.items():
                df[f'Home_{col_name}'] = values
        
        if away_stats:
            away_averages = self.calculate_average_stats(df, 'AwayTeam', away_stats)
            for col_name, values in away_averages.items():
                df[f'Away_{col_name}'] = values
        
        return df
    
    def create_context_features(self, df):
        """
        Crée les features contextuelles.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame d'entrée
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame avec features contextuelles
        """
        print("Creation des features contextuelles...")
        
        # Importance du match (basée sur la progression de la saison)
        if 'Date' in df.columns:
            try:
                df['Month'] = df['Date'].dt.month
                df['MatchWeek'] = df.groupby(df['Date'].dt.to_period('M')).cumcount() + 1
                max_week = df['MatchWeek'].max()
                df['MatchImportance'] = df['MatchWeek'] / max_week if max_week > 0 else 0.5
            except:
                df['MatchImportance'] = 0.5
        
        # Jours de repos
        if 'Date' in df.columns:
            df_sorted = df.sort_values('Date')
            df_sorted['DaysSinceLastHome'] = df_sorted.groupby('HomeTeam')['Date'].diff().dt.days.fillna(7)
            df_sorted['DaysSinceLastAway'] = df_sorted.groupby('AwayTeam')['Date'].diff().dt.days.fillna(7)
            df = df_sorted
        
        return df
    
    def encode_categorical_features(self, df):
        """
        Encode les features catégorielles.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame d'entrée
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame avec features encodées
        """
        print("Encodage des features categoriques...")
        
        # Encodage des équipes
        if 'HomeTeam' in df.columns:
            all_teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
            self.team_encoder.fit(all_teams)
            df['HomeTeam_encoded'] = self.team_encoder.transform(df['HomeTeam'])
            df['AwayTeam_encoded'] = self.team_encoder.transform(df['AwayTeam'])
        
        return df
    
    def select_features(self, df, target_col='ResultCode', k=15):
        """
        Sélectionne les meilleures features.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame d'entrée
        target_col : str
            Colonne cible
        k : int
            Nombre de features à sélectionner
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame avec features sélectionnées
        """
        if target_col not in df.columns:
            print(f"Target {target_col} non trouvee. Selection annulee.")
            return df
        
        # Préparer les données
        X = df.select_dtypes(include=[np.number])
        X = X.drop(columns=[target_col], errors='ignore')
        X = X.fillna(X.mean())
        
        y = df[target_col]
        
        # Sélection des features
        selector = SelectKBest(score_func=f_classif, k=min(k, len(X.columns)))
        X_selected = selector.fit_transform(X, y)
        
        # Récupérer les features sélectionnées
        self.selected_features = X.columns[selector.get_support()].tolist()
        
        print(f"Features selectionnees : {len(self.selected_features)}")
        print("Top features :")
        scores = selector.scores_[selector.get_support()]
        feature_scores = sorted(zip(self.selected_features, scores), key=lambda x: x[1], reverse=True)
        
        for i, (feature, score) in enumerate(feature_scores[:10], 1):
            print(f"  {i:2d}. {feature:25} : {score:.3f}")
        
        return df[self.selected_features + [target_col]]
    
    def normalize_features(self, df, feature_cols=None):
        """
        Normalise les features.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame d'entrée
        feature_cols : list, optional
            Liste des colonnes à normaliser
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame avec features normalisées
        """
        print("Normalisation des features...")
        
        if feature_cols is None:
            # Normaliser toutes les colonnes numériques sauf la target
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in numeric_cols if col != 'ResultCode']
        
        # Normalisation
        scaled_values = self.scaler.fit_transform(df[feature_cols])
        scaled_df = pd.DataFrame(scaled_values, columns=[f'{col}_scaled' for col in feature_cols])
        
        # Combiner avec le DataFrame original
        result_df = pd.concat([df, scaled_df], axis=1)
        
        return result_df
    
    def engineer_features(self, df, save_path=None):
        """
        Exécute le pipeline complet de feature engineering.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame d'entrée
        save_path : str, optional
            Chemin pour sauvegarder les features
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame avec toutes les features
        """
        print("\n" + "="*50)
        print("DEBUT DU FEATURE ENGINEERING")
        print("="*50)
        
        # 1. Features d'équipes
        df = self.create_team_features(df)
        
        # 2. Features contextuelles
        df = self.create_context_features(df)
        
        # 3. Encodage catégoriel
        df = self.encode_categorical_features(df)
        
        # 4. Sélection de features (si target disponible)
        if 'ResultCode' in df.columns:
            df = self.select_features(df)
        
        # 5. Normalisation
        df = self.normalize_features(df)
        
        # Sauvegarde
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df.to_csv(save_path, index=False)
            print(f"\nFeatures sauvegardees : {save_path}")
            
            # Sauvegarder la liste des features sélectionnées
            if self.selected_features:
                features_path = save_path.replace('.csv', '_features.txt')
                with open(features_path, 'w') as f:
                    for feature in self.selected_features:
                        f.write(f"{feature}\n")
                print(f"Liste des features sauvegardee : {features_path}")
        
        print("\n" + "="*50)
        print("FEATURE ENGINEERING TERMINE")
        print("="*50)
        print(f"Shape final : {df.shape}")
        print(f"Nouvelles features creees : {len(df.columns)}")
        
        return df
    
    def get_feature_summary(self, df):
        """
        Génère un résumé des features.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame d'entrée
            
        Returns:
        --------
        dict
            Résumé des features
        """
        summary = {
            'total_features': len(df.columns),
            'numeric_features': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(df.select_dtypes(include=['object']).columns),
            'selected_features': len(self.selected_features),
            'has_target': 'ResultCode' in df.columns
        }
        
        return summary