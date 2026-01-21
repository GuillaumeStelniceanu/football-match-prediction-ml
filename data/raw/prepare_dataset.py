

import pandas as pd
import numpy as np
import os

def load_individual_datasets():
    """
    Charge les 5 fichiers CSV des championnats
    """
    datasets = []
    championship_names = ['Ligue1', 'PremierLeague', 'Bundesliga', 'SerieA', 'LaLiga']
    
    for i, name in enumerate(championship_names):
        file_path = f"season-2223 ({i if i>0 else ''}).csv".replace(' ()', '')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['Championship'] = name
            datasets.append(df)
            print(f"✅ {name} chargé : {len(df)} matchs")
        else:
            print(f"⚠️  Fichier non trouvé : {file_path}")
    
    return datasets, championship_names

def clean_and_rename_data(df):
    """
    Nettoie et renomme les colonnes
    """
    # Renommer les colonnes principales
    df = df.rename(columns={
        'FTHG': 'HomeGoals',
        'FTAG': 'AwayGoals',
        'FTR': 'Result',
        'HS': 'HomeShots',
        'AS': 'AwayShots',
        'HST': 'HomeShotsOnTarget',
        'AST': 'AwayShotsOnTarget',
        'HC': 'HomeCorners',
        'AC': 'AwayCorners',
        'HY': 'HomeYellowCards',
        'AY': 'AwayYellowCards',
        'HR': 'HomeRedCards',
        'AR': 'AwayRedCards'
    })
    
    # Nettoyer les valeurs manquantes
    df = df.dropna(subset=['HomeGoals', 'AwayGoals'])
    
    return df

def create_features(df):
    """
    Crée les features pour l'analyse ML
    """
    # Variable cible
    df['GoalDiff'] = df['HomeGoals'] - df['AwayGoals']
    df['ResultCode'] = df['GoalDiff'].apply(
        lambda x: 1 if x > 0 else (0 if x == 0 else -1)
    )
    
    # Features de différences
    df['ShotDiff'] = df['HomeShots'] - df['AwayShots']
    df['ShotOnTargetDiff'] = df['HomeShotsOnTarget'] - df['AwayShotsOnTarget']
    df['CornerDiff'] = df['HomeCorners'] - df['AwayCorners']
    df['YellowCardDiff'] = df['HomeYellowCards'] - df['AwayYellowCards']
    df['RedCardDiff'] = df['HomeRedCards'] - df['AwayRedCards']
    
    # Features avancées
    df['TotalGoals'] = df['HomeGoals'] + df['AwayGoals']
    df['TotalShots'] = df['HomeShots'] + df['AwayShots']
    df['TotalCorners'] = df['HomeCorners'] + df['AwayCorners']
    
    # Taux de conversion
    df['HomeConversionRate'] = np.where(
        df['HomeShots'] > 0, 
        df['HomeGoals'] / df['HomeShots'] * 100, 
        0
    )
    df['AwayConversionRate'] = np.where(
        df['AwayShots'] > 0, 
        df['AwayGoals'] / df['AwayShots'] * 100, 
        0
    )
    
    return df

def prepare_final_dataset():
    """
    Pipeline principal de préparation des données
    """
    print("=" * 60)
    print("⚽ PRÉPARATION DU DATASET DE MATCHS DE FOOTBALL")
    print("=" * 60)
    
    # 1. Charger les datasets individuels
    datasets, championship_names = load_individual_datasets()
    
    if not datasets:
        print(" Aucun dataset trouvé. Vérifiez les fichiers CSV.")
        return None
    
    # 2. Combiner tous les datasets
    combined_df = pd.concat(datasets, ignore_index=True)
    print(f"\n📊 Dataset combiné : {len(combined_df)} matchs")
    
    # 3. Nettoyer et renommer
    combined_df = clean_and_rename_data(combined_df)
    print(f"📊 Après nettoyage : {len(combined_df)} matchs")
    
    # 4. Créer les features
    combined_df = create_features(combined_df)
    
    # 5. Sélectionner les colonnes finales
    final_columns = [
        'Date', 'HomeTeam', 'AwayTeam', 'Championship',
        'HomeGoals', 'AwayGoals', 'GoalDiff', 'ResultCode',
        'HomeShots', 'AwayShots', 'ShotDiff',
        'HomeShotsOnTarget', 'AwayShotsOnTarget', 'ShotOnTargetDiff',
        'HomeCorners', 'AwayCorners', 'CornerDiff',
        'HomeYellowCards', 'AwayYellowCards', 'YellowCardDiff',
        'HomeRedCards', 'AwayRedCards', 'RedCardDiff',
        'TotalGoals', 'TotalShots', 'TotalCorners',
        'HomeConversionRate', 'AwayConversionRate'
    ]
    
    # Garder seulement les colonnes disponibles
    available_columns = [col for col in final_columns if col in combined_df.columns]
    final_df = combined_df[available_columns]
    
    # 6. Sauvegarder
    output_path = "data/raw/matches.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)
    
    print(f"\n Dataset final sauvegardé dans : {output_path}")
    print(f" Dimensions : {final_df.shape}")
    print(f" Championnats : {final_df['Championship'].unique()}")
    print(f" Distribution des résultats :")
    print(final_df['ResultCode'].value_counts().sort_index())
    print("\n" + "=" * 60)
    
    return final_df

if __name__ == "__main__":
    # Exécuter la préparation
    df_final = prepare_final_dataset()
    
    if df_final is not None:
        # Aperçu des données
        print("\n🔍 APERÇU DU DATASET FINAL :")
        print(df_final.head())
        
        # Statistiques descriptives
        print("\n STATISTIQUES DESCRIPTIVES :")
        print(df_final[['HomeGoals', 'AwayGoals', 'GoalDiff', 
                       'ShotDiff', 'CornerDiff']].describe())