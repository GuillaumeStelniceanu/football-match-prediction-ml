"""
Script principal du projet
"""

def main():
    print("=" * 60)
    print("⚽ PROJET DE PRÉDICTION DE MATCHS DE FOOTBALL")
    print("=" * 60)
    
    # 1. Préparation des données
    print("\n1️⃣  PRÉPARATION DES DONNÉES")
    try:
        # Essayer de préparer les données si le fichier n'existe pas
        import os
        if not os.path.exists('data/raw/matches.csv'):
            print("   ⚠️  Fichier matches.csv non trouvé")
            print("   🔧 Exécution du script de préparation...")
            try:
                exec(open('data/raw/prepare_dataset.py').read())
            except:
                print("   ❌ Échec de préparation des données")
                print("   💡 Assurez-vous d'avoir les fichiers CSV dans le dossier")
                return
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return
    
    # 2. Prétraitement
    print("\n2️⃣  PRÉTRAITEMENT")
    try:
        from src.preprocessing import load_and_preprocess
        data = load_and_preprocess()
        if data is None:
            return
        X_train, X_test, y_train, y_test, feature_names = data
    except Exception as e:
        print(f"   ❌ Erreur de prétraitement: {e}")
        return
    
    # 3. Entraînement
    print("\n3️⃣  ENTRAÎNEMENT")
    try:
        from src.model_training import train_models, save_models
        models = train_models(X_train, y_train)
        if models:
            save_models(models, feature_names)
        else:
            print("   ❌ Aucun modèle entraîné")
            return
    except Exception as e:
        print(f"   ❌ Erreur d'entraînement: {e}")
        return
    
    # 4. Évaluation
    print("\n4️⃣  ÉVALUATION")
    try:
        from src.evaluation import evaluate_models
        results = evaluate_models(models, X_test, y_test)
        
        # Résumé
        print("\n" + "=" * 60)
        print("🎯 RÉSUMÉ DES PERFORMANCES")
        print("=" * 60)
        for name, accuracy in results.items():
            print(f"   {name:20} : {accuracy:.3f}")
        
        # Meilleur modèle
        best_model = max(results, key=results.get)
        print(f"\n🏆 MEILLEUR MODÈLE: {best_model} ({results[best_model]:.3f})")
        
    except Exception as e:
        print(f"   ❌ Erreur d'évaluation: {e}")
        return
    
    print("\n" + "=" * 60)
    print("✅ PROJET TERMINÉ AVEC SUCCÈS !")
    print("=" * 60)
    print("\n📁 FICHIERS CRÉÉS:")
    print("   • data/raw/matches.csv - Données combinées")
    print("   • models/*.pkl - Modèles sauvegardés")
    print("   • visuals/*.png - Visualisations")
    print("\n🚀 Projet prêt pour GitHub !")

if __name__ == "__main__":
    main()
