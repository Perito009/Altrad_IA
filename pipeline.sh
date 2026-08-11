#!/bin/bash

# Rediriger stdout vers un fichier log
exec > >(tee -a pipeline.log)

# Rediriger stderr vers le même fichier log
exec 2>&1

echo "Début de la pipeline ETL"
date +"%Y-%m-%d %H:%M:%S"

# Étape 1: Nettoyer le CSV avec refactorCSV.py
if python3 refactorCSV.py; then
    echo "Étape 1 terminée : succès"
else
    echo "Étape 1 échouée. Vérifiez les logs pour plus de détails."
    exit 1
fi

# Étape 2: Exécuter ETL pour nettoyer les données
if python3 ETL.py; then
    echo "Étape 2 terminée : succès"
else
    echo "Étape 2 échouée. Vérifiez les logs pour plus de détails."
    exit 1
fi

# Étape 3: Charger les données dans la base de données avec LoaderDB.py
if python3 LoaderDB.py; then
    echo "Étape 3 terminée : succès"
else
    echo "Étape 3 échouée. Vérifiez les logs pour plus de détails."
    exit 1
fi

echo "Fin de la pipeline ETL"
date +"%Y-%m-%d %H:%M:%S"