# Altrad_IA

## Système d'analyse et de recommandation des ressources SharePoint

---

## 1. Présentation du projet

**Altrad_IA** est un projet de Data Engineering, Data Analysis et Machine Learning réalisé autour des données d'utilisation des ressources SharePoint.

L'objectif du projet est de transformer des données brutes provenant de l'environnement SharePoint en données exploitables afin de :

- nettoyer et structurer les données ;
- analyser l'utilisation des ressources SharePoint ;
- identifier les utilisateurs et leurs ressources ;
- calculer la popularité des ressources ;
- identifier des utilisateurs présentant des comportements similaires ;
- générer des recommandations personnalisées ;
- évaluer les performances du système de recommandation ;
- mettre à disposition les résultats dans une interface Streamlit ;
- stocker les données structurées dans une base SQLite.

Le projet constitue ainsi une chaîne complète allant de la donnée brute jusqu'à une application permettant d'exploiter les résultats du modèle.

---

# 2. Contexte de l'entreprise

Le projet est réalisé dans le contexte de l'entreprise **Altrad Services France**, appartenant au groupe **Altrad**.

L'entreprise utilise de nombreux espaces SharePoint permettant aux collaborateurs d'accéder à différentes ressources :

- sites ;
- sous-sites ;
- listes ;
- bibliothèques ;
- documents ;
- espaces de travail ;
- ressources liées aux différents services et projets.

Le volume important de ressources et d'utilisateurs rend difficile l'identification des ressources pertinentes pour chaque utilisateur.

Le projet Altrad_IA vise donc à exploiter les données disponibles afin de mieux comprendre les usages et de proposer un système de recommandation basé sur les comportements observés.

---

# 3. Problématique

Les données SharePoint initiales sont principalement constituées d'informations associant :

- un site ;
- un sous-site ;
- une liste ;
- une bibliothèque ;
- un utilisateur.

La problématique consiste à répondre à la question suivante :

> Comment exploiter les données d'utilisation de SharePoint afin d'identifier les ressources pertinentes pour les utilisateurs et de proposer des recommandations personnalisées ?

Pour répondre à cette problématique, plusieurs étapes ont été mises en place :

1. préparation des données ;
2. stockage structuré ;
3. analyse des interactions utilisateurs/ressources ;
4. calcul de similarité entre utilisateurs ;
5. analyse de popularité ;
6. génération des recommandations ;
7. évaluation du modèle ;
8. visualisation dans une application Streamlit.

---

# 4. Objectifs

## 4.1 Objectif principal

Construire un système capable d'exploiter les interactions SharePoint afin de proposer des ressources susceptibles d'intéresser un utilisateur.

## 4.2 Objectifs techniques

Le projet doit permettre de :

- charger les données CSV ;
- contrôler leur structure ;
- nettoyer les données ;
- supprimer les utilisateurs invalides ;
- supprimer les doublons ;
- construire une représentation unique des ressources ;
- construire les interactions utilisateurs/ressources ;
- stocker les données préparées ;
- alimenter une base SQLite ;
- calculer les similarités entre utilisateurs ;
- calculer la popularité des ressources ;
- générer des recommandations ;
- mesurer la qualité des recommandations ;
- présenter les résultats avec Streamlit.

---

# 5. Architecture du projet

L'architecture générale du projet peut être représentée de la manière suivante :

```text
                         CSV SharePoint
                               |
                               v
                    +---------------------+
                    | preprocessing.py    |
                    | Nettoyage /         |
                    | transformation      |
                    +----------+----------+
                               |
                 +-------------+-------------+
                 |                           |
                 v                           v
       interactions.parquet          resources.parquet
                 |                           |
                 |                           |
                 v                           v
          +-------------+             +-------------+
          | similarity  |             | popularity  |
          |     .py     |             |     .py     |
          +------+------+             +------+------+
                 |                           |
                 v                           v
       user_neighbors.parquet       resource_popularity.parquet
                 |                           |
                 +-------------+-------------+
                               |
                               v
                       recommender.py
                               |
                               v
                      Recommandations
                               |
                               v
                        evaluation.py
                               |
                               v
                       Métriques ML
                               |
                               v
                        Application
                         Streamlit
                               |
                               v
                      Interface utilisateur