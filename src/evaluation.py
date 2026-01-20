"""
Évaluation des modèles
"""

import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate_models(models, X_test, y_test):
    """Évalue tous les modèles"""
    print("\n📊 ÉVALUATION DES MODÈLES")
    print("-" * 40)
    
    results = {}
    
    for name, model in models.items():
        print(f"\n🔍 Évaluation {name}...")
        
        # Prédictions
        y_pred = model.predict(X_test)
        
        # Métriques
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        
        print(f"   Accuracy: {accuracy:.3f}")
        
        # Rapport de classification
        print("   Rapport:")
        report = classification_report(y_test, y_pred, target_names=['Extérieur', 'Nul', 'Domicile'])
        for line in report.split('\n'):
            print(f"     {line}")
        
        # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        
        # Visualisation
        os.makedirs('visuals', exist_ok=True)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Extérieur', 'Nul', 'Domicile'],
                   yticklabels=['Extérieur', 'Nul', 'Domicile'])
        plt.title(f'Matrice de Confusion - {name}')
        plt.xlabel('Prédictions')
        plt.ylabel('Valeurs Réelles')
        plt.tight_layout()
        plt.savefig(f'visuals/confusion_{name}.png', dpi=150)
        plt.close()
        print(f"   📈 Matrice sauvegardée: visuals/confusion_{name}.png")
    
    return results
