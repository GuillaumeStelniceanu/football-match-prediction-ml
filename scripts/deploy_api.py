
#!/usr/bin/env python3
# Script de déploiement pour l'API de prédiction

import pickle
import json
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

# Charger le package de déploiement
try:
    with open('deployment_package.pkl', 'rb') as f:
        deployment_data = pickle.load(f)
    print(f"Modele charge : {deployment_data['model_type']} v{deployment_data['model_version']}")
except:
    print("Erreur lors du chargement du modele")
    deployment_data = None

@app.route('/health', methods=['GET'])
def health_check():
    if deployment_data:
        return jsonify({'status': 'healthy', 'model_version': deployment_data['model_version']})
    else:
        return jsonify({'status': 'unhealthy', 'error': 'Modele non charge'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not deployment_data:
            return jsonify({'error': 'Modele non disponible'}), 500

        data = request.json

        # Validation des données
        if 'HomeTeam' not in data or 'AwayTeam' not in data:
            return jsonify({'error': 'HomeTeam et AwayTeam sont requis'}), 400

        # Ici, normalement vous chargeriez le modèle et feriez la prédiction
        # Pour cet exemple, nous retournons une prédiction factice

        response = {
            'match': f"{data['HomeTeam']} vs {data['AwayTeam']}",
            'prediction': 'Victoire Domicile',
            'confidence': 0.75,
            'probabilities': {'Domicile': 0.75, 'Nul': 0.15, 'Exterieur': 0.10}
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
