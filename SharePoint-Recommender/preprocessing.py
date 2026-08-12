"""
===============================================================================
                    SHAREPOINT RECOMMENDER
                       PREPROCESSING PIPELINE
===============================================================================

Objectif
--------
Préparer le dataset SharePoint afin de construire un système de
recommandation basé sur les interactions utilisateurs / ressources.

Dataset source
--------------
Data/Cleaned_BD_Sharepoint.csv

Colonnes attendues dans le CSV
------------------------------
Site
Sous-Site
Liste
Bibliothèque
Users

Colonnes utilisées après normalisation
--------------------------------------
site
sous-site
liste
bibliothèque
users

Sorties
-------
processed/clean_data.parquet
processed/user_resource.parquet

Matrice utilisateur × ressource
--------------------------------
Les lignes représentent les utilisateurs.

Les colonnes représentent les ressources SharePoint.

Une valeur de 1 signifie :

    l'utilisateur a utilisé / possède / consulte cette ressource.

Une valeur de 0 signifie :

    aucune interaction connue.

Cette matrice sera ensuite utilisée par similarity.py pour calculer
la similarité entre utilisateurs.

===============================================================================
"""


# =============================================================================
# IMPORTS
# =============================================================================

from pathlib import Path
import logging

import pandas as pd
import numpy as np


# =============================================================================
# CONFIGURATION
# =============================================================================

# -------------------------------------------------------------------------
# Répertoire racine du projet
# -------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent


# -------------------------------------------------------------------------
# Fichier CSV source
# -------------------------------------------------------------------------

DATA_PATH = (
    BASE_DIR
    / "Data"
    / "Cleaned_BD_Sharepoint.csv"
)


# -------------------------------------------------------------------------
# Répertoire de sortie
# -------------------------------------------------------------------------

PROCESSED_DIR = (
    BASE_DIR
    / "processed"
)


# -------------------------------------------------------------------------
# Fichiers générés
# -------------------------------------------------------------------------

CLEAN_DATA_PATH = (
    PROCESSED_DIR
    / "clean_data.parquet"
)

USER_RESOURCE_PATH = (
    PROCESSED_DIR
    / "user_resource.parquet"
)


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s - "
        "%(levelname)s - "
        "%(message)s"
    ),
)


logger = logging.getLogger(__name__)


# =============================================================================
# COLONNES ATTENDUES
# =============================================================================

REQUIRED_COLUMNS = [
    "site",
    "sous-site",
    "liste",
    "bibliothèque",
    "users",
]


# =============================================================================
# CHARGEMENT DU DATASET
# =============================================================================

def load_dataset():
    """
    Charge le fichier CSV SharePoint.

    Le CSV fourni utilise une virgule comme séparateur.

    Returns
    -------
    pandas.DataFrame
        Dataset brut.
    """

    logger.info(
        "Chargement du dataset SharePoint."
    )

    logger.info(
        f"Fichier source : {DATA_PATH}"
    )


    # -------------------------------------------------------------------------
    # Vérification de l'existence du fichier
    # -------------------------------------------------------------------------

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"\n"
            f"Fichier CSV introuvable :\n"
            f"{DATA_PATH}\n\n"
            f"Vérifie que le fichier existe dans :\n"
            f"Data/Cleaned_BD_Sharepoint.csv"
        )


    # -------------------------------------------------------------------------
    # Lecture du CSV
    # -------------------------------------------------------------------------

    try:

        df = pd.read_csv(
            DATA_PATH,
            sep=",",
            encoding="utf-8-sig",
        )

    except UnicodeDecodeError:

        logger.warning(
            "UTF-8 non disponible. "
            "Nouvelle tentative avec latin-1."
        )

        df = pd.read_csv(
            DATA_PATH,
            sep=",",
            encoding="latin-1",
        )


    logger.info(
        f"Dataset chargé : "
        f"{df.shape[0]} lignes, "
        f"{df.shape[1]} colonnes."
    )


    return df


# =============================================================================
# NORMALISATION DES COLONNES
# =============================================================================

def normalize_column_names(df):
    """
    Normalise les noms de colonnes.

    Exemple :

        'Site'              -> 'site'
        'Sous-Site'         -> 'sous-site'
        'Bibliothèque '     -> 'bibliothèque'
        'Users'             -> 'users'

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset source.

    Returns
    -------
    pandas.DataFrame
        Dataset avec colonnes normalisées.
    """

    logger.info(
        "Normalisation des noms de colonnes."
    )


    # Suppression des espaces autour des noms.
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )


    # Conversion en minuscules.
    df.columns = (
        df.columns
        .str.lower()
    )


    logger.info(
        f"Colonnes après normalisation : "
        f"{df.columns.tolist()}"
    )


    return df


# =============================================================================
# VALIDATION DU SCHEMA
# =============================================================================

def validate_schema(df):
    """
    Vérifie que toutes les colonnes nécessaires
    sont présentes.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset à vérifier.

    Raises
    ------
    ValueError
        Si des colonnes sont absentes.
    """

    logger.info(
        "Validation du schéma du dataset."
    )


    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]


    if missing_columns:

        raise ValueError(
            "Colonnes manquantes : "
            + ", ".join(missing_columns)
            + "\n\n"
            + "Colonnes trouvées : "
            + ", ".join(df.columns.tolist())
        )


    logger.info(
        "Validation du schéma réussie."
    )


# =============================================================================
# NETTOYAGE DES VALEURS
# =============================================================================

def clean_values(df):
    """
    Nettoie les valeurs du dataset.

    Opérations :

    - suppression des espaces ;
    - conversion en chaînes ;
    - conversion des valeurs '0' en valeurs manquantes ;
    - conversion des chaînes vides en valeurs manquantes ;
    - suppression des lignes sans utilisateur.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
    """

    logger.info(
        "Nettoyage des valeurs."
    )


    df = df.copy()


    # -------------------------------------------------------------------------
    # Nettoyage des colonnes textuelles
    # -------------------------------------------------------------------------

    for column in REQUIRED_COLUMNS:

        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
        )


    # -------------------------------------------------------------------------
    # Remplacement des valeurs vides
    # -------------------------------------------------------------------------

    df = df.replace(
        {
            "": pd.NA,
            "nan": pd.NA,
            "None": pd.NA,
            "NULL": pd.NA,
        }
    )


    # -------------------------------------------------------------------------
    # Dans ton dataset, 0 représente généralement
    # une information non renseignée.
    #
    # On le transforme uniquement pour les dimensions
    # SharePoint, pas pour les utilisateurs.
    # -------------------------------------------------------------------------

    dimension_columns = [
        "site",
        "sous-site",
        "liste",
        "bibliothèque",
    ]


    for column in dimension_columns:

        df[column] = df[column].replace(
            "0",
            pd.NA
        )


    # -------------------------------------------------------------------------
    # Suppression des lignes sans utilisateur
    # -------------------------------------------------------------------------

    before = len(df)


    df = df.dropna(
        subset=["users"]
    )


    after = len(df)


    removed = before - after


    if removed > 0:

        logger.info(
            f"{removed} lignes supprimées "
            f"car aucun utilisateur n'était renseigné."
        )


    # -------------------------------------------------------------------------
    # Suppression des doublons exacts
    # -------------------------------------------------------------------------

    before = len(df)


    df = df.drop_duplicates()


    after = len(df)


    removed = before - after


    if removed > 0:

        logger.info(
            f"{removed} doublons exacts supprimés."
        )


    # -------------------------------------------------------------------------
    # Réinitialisation de l'index
    # -------------------------------------------------------------------------

    df = df.reset_index(
        drop=True
    )


    logger.info(
        f"Dataset après nettoyage : "
        f"{len(df)} lignes."
    )


    return df


# =============================================================================
# CREATION DE LA RESSOURCE
# =============================================================================

def create_resource_column(df):
    """
    Crée un identifiant unique pour chaque ressource SharePoint.

    Une ressource est définie par :

        Site
        +
        Sous-site
        +
        Liste
        +
        Bibliothèque

    Exemple :

        Altrad S fr : Home
        |
        Altrad S fr : Home
        |
        BDD-RH Personnels
        |
        Document de la collection de sites

    devient :

        Altrad S fr : Home |
        Altrad S fr : Home |
        BDD-RH Personnels |
        Document de la collection de sites

    Cette colonne est utilisée comme identifiant de ressource
    dans le système de recommandation.
    """

    logger.info(
        "Création des identifiants de ressources."
    )


    df = df.copy()


    # -------------------------------------------------------------------------
    # Remplacement temporaire des valeurs manquantes
    # -------------------------------------------------------------------------

    resource_columns = [
        "site",
        "sous-site",
        "liste",
        "bibliothèque",
    ]


    for column in resource_columns:

        df[column] = (
            df[column]
            .fillna("Non renseigné")
        )


    # -------------------------------------------------------------------------
    # Création de l'identifiant ressource
    # -------------------------------------------------------------------------

    df["resource"] = (
        df["site"]
        + " | "
        + df["sous-site"]
        + " | "
        + df["liste"]
        + " | "
        + df["bibliothèque"]
    )


    # -------------------------------------------------------------------------
    # Nettoyage final de l'identifiant
    # -------------------------------------------------------------------------

    df["resource"] = (
        df["resource"]
        .str.strip()
    )


    logger.info(
        f"Nombre de ressources uniques : "
        f"{df['resource'].nunique()}"
    )


    return df


# =============================================================================
# CREATION DE LA MATRICE UTILISATEUR × RESSOURCE
# =============================================================================

def create_user_resource_matrix(df):
    """
    Construit la matrice utilisateur × ressource.

    Exemple :

    +-----------+-----------+-----------+-----------+
    | users     | Resource A| Resource B| Resource C|
    +-----------+-----------+-----------+-----------+
    | Yann      |     1     |     0     |     1     |
    | Marie     |     0     |     1     |     1     |
    | Paul      |     1     |     1     |     0     |
    +-----------+-----------+-----------+-----------+

    1 = interaction connue
    0 = aucune interaction connue

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
    """

    logger.info(
        "Construction de la matrice "
        "utilisateur × ressource."
    )


    # -------------------------------------------------------------------------
    # Pivot
    # -------------------------------------------------------------------------

    matrix = pd.crosstab(
        df["users"],
        df["resource"],
    )


    # -------------------------------------------------------------------------
    # Transformation en binaire
    #
    # Si un utilisateur apparaît plusieurs fois
    # pour la même ressource, on garde simplement 1.
    # -------------------------------------------------------------------------

    matrix = (
        matrix
        .clip(upper=1)
        .astype("int8")
    )


    # -------------------------------------------------------------------------
    # Tri
    # -------------------------------------------------------------------------

    matrix = matrix.sort_index()

    matrix = matrix.sort_index(
        axis=1
    )


    logger.info(
        "Matrice utilisateur × ressource créée."
    )


    logger.info(
        f"Nombre d'utilisateurs : "
        f"{matrix.shape[0]}"
    )


    logger.info(
        f"Nombre de ressources : "
        f"{matrix.shape[1]}"
    )


    logger.info(
        f"Taille de la matrice : "
        f"{matrix.shape[0]} × {matrix.shape[1]}"
    )


    return matrix


# =============================================================================
# STATISTIQUES DU DATASET
# =============================================================================

def generate_statistics(
    df,
    user_resource_matrix,
):
    """
    Affiche quelques statistiques utiles
    sur le dataset préparé.
    """

    logger.info(
        "Calcul des statistiques."
    )


    number_rows = len(df)

    number_users = (
        df["users"]
        .nunique()
    )

    number_resources = (
        df["resource"]
        .nunique()
    )


    # -------------------------------------------------------------------------
    # Densité de la matrice
    # -------------------------------------------------------------------------

    total_possible_interactions = (
        user_resource_matrix.shape[0]
        * user_resource_matrix.shape[1]
    )


    total_interactions = (
        user_resource_matrix
        .to_numpy()
        .sum()
    )


    if total_possible_interactions > 0:

        density = (
            total_interactions
            / total_possible_interactions
        )

    else:

        density = 0


    logger.info(
        "======================================================================"
    )

    logger.info(
        "STATISTIQUES DU DATASET"
    )

    logger.info(
        "======================================================================"
    )

    logger.info(
        f"Nombre de lignes              : {number_rows}"
    )

    logger.info(
        f"Nombre d'utilisateurs         : {number_users}"
    )

    logger.info(
        f"Nombre de ressources          : {number_resources}"
    )

    logger.info(
        f"Nombre d'interactions        : {int(total_interactions)}"
    )

    logger.info(
        f"Densité de la matrice         : "
        f"{density:.6f}"
    )

    logger.info(
        "======================================================================"
    )


# =============================================================================
# SAUVEGARDE
# =============================================================================

def save_processed_data(
    df,
    user_resource_matrix,
):
    """
    Sauvegarde les données préparées au format Parquet.

    Fichiers :

        processed/clean_data.parquet

        processed/user_resource.parquet
    """

    logger.info(
        "Sauvegarde des données préparées."
    )


    # -------------------------------------------------------------------------
    # Création du dossier processed
    # -------------------------------------------------------------------------

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # -------------------------------------------------------------------------
    # Sauvegarde du dataset nettoyé
    # -------------------------------------------------------------------------

    df.to_parquet(
        CLEAN_DATA_PATH,
        index=False
    )


    logger.info(
        f"Dataset nettoyé sauvegardé : "
        f"{CLEAN_DATA_PATH}"
    )


    # -------------------------------------------------------------------------
    # Sauvegarde de la matrice
    # -------------------------------------------------------------------------

    user_resource_matrix.to_parquet(
        USER_RESOURCE_PATH
    )


    logger.info(
        f"Matrice utilisateur × ressource sauvegardée : "
        f"{USER_RESOURCE_PATH}"
    )


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def run_preprocessing():
    """
    Exécute l'intégralité du pipeline de preprocessing.

    Étapes :

        1. Chargement
        2. Normalisation des colonnes
        3. Validation
        4. Nettoyage
        5. Création des ressources
        6. Création de la matrice utilisateur × ressource
        7. Statistiques
        8. Sauvegarde

    Returns
    -------
    dict
        Résultats du preprocessing.
    """

    logger.info(
        "======================================================================"
    )

    logger.info(
        "DÉMARRAGE DU PIPELINE SHAREPOINT"
    )

    logger.info(
        "======================================================================"
    )


    # =========================================================================
    # 1. CHARGEMENT
    # =========================================================================

    df = load_dataset()


    # =========================================================================
    # 2. NORMALISATION DES COLONNES
    # =========================================================================

    df = normalize_column_names(
        df
    )


    # =========================================================================
    # 3. VALIDATION DU SCHEMA
    # =========================================================================

    validate_schema(
        df
    )


    # =========================================================================
    # 4. NETTOYAGE
    # =========================================================================

    df = clean_values(
        df
    )


    # =========================================================================
    # 5. CREATION DES RESSOURCES
    # =========================================================================

    df = create_resource_column(
        df
    )


    # =========================================================================
    # 6. MATRICE UTILISATEUR × RESSOURCE
    # =========================================================================

    user_resource_matrix = (
        create_user_resource_matrix(
            df
        )
    )


    # =========================================================================
    # 7. STATISTIQUES
    # =========================================================================

    generate_statistics(
        df,
        user_resource_matrix
    )


    # =========================================================================
    # 8. SAUVEGARDE
    # =========================================================================

    save_processed_data(
        df,
        user_resource_matrix
    )


    # =========================================================================
    # FIN
    # =========================================================================

    logger.info(
        "======================================================================"
    )

    logger.info(
        "PREPROCESSING TERMINÉ AVEC SUCCÈS"
    )

    logger.info(
        "======================================================================"
    )


    return {
        "clean_data": df,
        "user_resource_matrix": user_resource_matrix,
    }


# =============================================================================
# EXECUTION
# =============================================================================

if __name__ == "__main__":

    try:

        results = run_preprocessing()


        # ---------------------------------------------------------------------
        # Résumé final dans le terminal
        # ---------------------------------------------------------------------

        df_clean = results[
            "clean_data"
        ]

        matrix = results[
            "user_resource_matrix"
        ]


        print()
        print(
            "============================================================"
        )

        print(
            "✅ PREPROCESSING TERMINÉ"
        )

        print(
            "============================================================"
        )

        print(
            f"📄 Lignes nettoyées      : {len(df_clean):,}"
        )

        print(
            f"👥 Utilisateurs          : "
            f"{matrix.shape[0]:,}"
        )

        print(
            f"📚 Ressources            : "
            f"{matrix.shape[1]:,}"
        )

        print()
        print(
            "📁 Fichiers générés :"
        )

        print(
            f"   → {CLEAN_DATA_PATH}"
        )

        print(
            f"   → {USER_RESOURCE_PATH}"
        )

        print(
            "============================================================"
        )


    except Exception as error:

        logger.exception(
            "Erreur pendant le preprocessing."
        )

        print()
        print(
            "❌ ERREUR DU PREPROCESSING"
        )

        print(
            str(error)
        )

        raise