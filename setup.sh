#!/bin/bash
# Créer et activer l'environnement virtuel
python -m venv venv
source venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt

echo "✅ Installation terminée!"
echo "Pour activer l'environnement: source venv/bin/activate"
