"""
Validation des données de matchs
"""

import pandas as pd
import numpy as np

class DataValidator:
    """Classe pour valider les données de matchs."""
    
    def __init__(self, df):
        """
        Initialise le validateur.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame à valider
        """
        self.df = df
        self.validation_results = {}
    
    def validate_numeric_ranges(self):
        """
        Valide les plages de valeurs numériques.
        
        Returns:
        --------
        dict
            Résultats de validation
        """
        results = {}
        
        # Plages attendues pour différentes variables
        expected_ranges = {
            'HomeGoals': (0, 10),
            'AwayGoals': (0, 10),
            'HomeShots': (0, 40),
            'AwayShots': (0, 40),
            'HomeYellowCards': (0, 10),
            'AwayYellowCards': (0, 10)
        }
        
        for col, (min_val, max_val) in expected_ranges.items():
            if col in self.df.columns:
                violations = self.df[(self.df[col] < min_val) | (self.df[col] > max_val)]
                results[col] = {
                    'expected_min': min_val,
                    'expected_max': max_val,
                    'actual_min': self.df[col].min(),
                    'actual_max': self.df[col].max(),
                    'violations': len(violations),
                    'is_valid': len(violations) == 0
                }
        
        self.validation_results['numeric_ranges'] = results
        return results
    
    def validate_target_distribution(self):
        """
        Valide la distribution de la target.
        
        Returns:
        --------
        dict
            Résultats de validation
        """
        results = {}
        
        if 'ResultCode' in self.df.columns:
            distribution = self.df['ResultCode'].value_counts()
            total = len(self.df)
            
            results['distribution'] = distribution.to_dict()
            results['percentages'] = (distribution / total * 100).to_dict()
            results['is_balanced'] = all(20 < p < 50 for p in results['percentages'].values())
        
        self.validation_results['target_distribution'] = results
        return results
    
    def validate_missing_values(self, threshold=0.1):
        """
        Valide les valeurs manquantes.
        
        Parameters:
        -----------
        threshold : float
            Seuil acceptable pour les valeurs manquantes
            
        Returns:
        --------
        dict
            Résultats de validation
        """
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        
        results = {
            'missing_counts': missing[missing > 0].to_dict(),
            'missing_percentages': missing_pct[missing_pct > 0].to_dict(),
            'problematic_columns': missing_pct[missing_pct > threshold].index.tolist(),
            'is_valid': all(missing_pct <= threshold)
        }
        
        self.validation_results['missing_values'] = results
        return results
    
    def validate_correlations(self):
        """
        Valide les corrélations entre variables.
        
        Returns:
        --------
        dict
            Résultats de validation
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        corr_matrix = self.df[numeric_cols].corr()
        
        # Identifier les corrélations fortes
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr = abs(corr_matrix.iloc[i, j])
                if corr > 0.8:
                    strong_correlations.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': corr
                    })
        
        results = {
            'strong_correlations': strong_correlations,
            'high_correlation_count': len(strong_correlations),
            'has_high_correlation': len(strong_correlations) > 0
        }
        
        self.validation_results['correlations'] = results
        return results
    
    def run_all_validations(self):
        """
        Exécute toutes les validations.
        
        Returns:
        --------
        dict
            Résultats complets de validation
        """
        print("\n" + "="*50)
        print("VALIDATION DES DONNEES")
        print("="*50)
        
        self.validate_numeric_ranges()
        self.validate_target_distribution()
        self.validate_missing_values()
        self.validate_correlations()
        
        # Résumé
        is_valid = all([
            all(r['is_valid'] for r in self.validation_results['numeric_ranges'].values()),
            self.validation_results['missing_values']['is_valid'],
            not self.validation_results['correlations']['has_high_correlation']
        ])
        
        self.validation_results['overall_valid'] = is_valid
        
        print(f"\nValidation terminee. Donnees valides : {is_valid}")
        
        if not is_valid:
            print("\nProblemes identifies :")
            for check_name, results in self.validation_results.items():
                if check_name != 'overall_valid':
                    if isinstance(results, dict) and 'is_valid' in results:
                        if not results['is_valid']:
                            print(f"  - {check_name}")
        
        return self.validation_results
    
    def get_validation_report(self):
        """
        Génère un rapport de validation.
        
        Returns:
        --------
        str
            Rapport formaté
        """
        report = []
        report.append("="*60)
        report.append("RAPPORT DE VALIDATION DES DONNEES")
        report.append("="*60)
        
        for check_name, results in self.validation_results.items():
            if check_name != 'overall_valid':
                report.append(f"\n{check_name.upper().replace('_', ' ')}:")
                report.append("-" * 40)
                
                if isinstance(results, dict):
                    for key, value in results.items():
                        if key not in ['is_valid', 'problematic_columns', 'strong_correlations']:
                            report.append(f"  {key}: {value}")
        
        report.append(f"\n{'='*60}")
        report.append(f"VALIDATION GLOBALE: {self.validation_results.get('overall_valid', 'N/A')}")
        report.append("="*60)
        
        return "\n".join(report)