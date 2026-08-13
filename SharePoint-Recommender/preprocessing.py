"""
===============================================================================
                    SHAREPOINT RECOMMENDER
                       PREPROCESSING ML
===============================================================================

Ce fichier intervient APRES le nettoyage ETL.

Le nettoyage est déjà réalisé par :

    1. ETL.py
    2. refactorCSV.py

Ce module prépare uniquement les données pour le Machine Learning.

Pipeline :

    BD_Sharepoint_Clean.csv
            |
            v
    Validation des colonnes
            |
            v
    Suppression des utilisateurs invalides
            |
            v
    Création d'une ressource SharePoint
            |
            v
    Suppression des doublons
            |
            v
    Création des interactions
            |
            v
    Données prêtes pour le recommender

===============================================================================
"""

from pathlib import Path

import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_PATH = Path(
    "data/BD_Sharepoint_Clean.csv"
)

PROCESSED_PATH = Path(
    "processed"
)

PROCESSED_PATH.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# COLONNES ATTENDUES
# =============================================================================

EXPECTED_COLUMNS = [
    "site",
    "sous-site",
    "liste",
    "bibliothèque",
    "users",
]


# =============================================================================
# UTILISATEURS INVALIDES
# =============================================================================

INVALID_USERS = {
    "aucun",
    "pas de droit",
    "inconnu",
    "",
    "nan",
    "none",
}


# =============================================================================
# CHARGEMENT
# =============================================================================

def load_data():
    """
    Charge le CSV déjà nettoyé.

    Aucun nettoyage métier n'est réalisé ici.
    """

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"""
Fichier introuvable :

    {DATA_PATH}

Place BD_Sharepoint_Clean.csv dans :

    data/
"""
        )

    df = pd.read_csv(
        DATA_PATH,
        encoding="utf-8"
    )

    print(
        f"CSV chargé : {len(df):,} lignes"
    )

    return df


# =============================================================================
# VALIDATION
# =============================================================================

def validate_data(df):
    """
    Vérifie que toutes les colonnes nécessaires
    sont présentes.
    """

    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Colonnes manquantes : "
            + ", ".join(
                missing_columns
            )
        )

    print(
        "✓ Structure du CSV valide"
    )


# =============================================================================
# NORMALISATION MINIMALE
# =============================================================================

def normalize_columns(df):
    """
    Effectue uniquement une normalisation technique.

    Le nettoyage métier a déjà été réalisé par l'ETL.

    On supprime notamment les espaces inutiles
    autour des valeurs.
    """

    df = df.copy()

    columns = [
        "site",
        "sous-site",
        "liste",
        "bibliothèque",
        "users",
    ]

    for column in columns:

        df[column] = (
            df[column]
            .fillna("aucun")
            .astype(str)
            .str.strip()
        )

    return df


# =============================================================================
# FILTRAGE DES UTILISATEURS
# =============================================================================

def filter_valid_users(df):
    """
    Supprime les lignes qui ne correspondent
    pas à un véritable utilisateur.

    Exemple :

        users = aucun

    ne doit PAS devenir un utilisateur du modèle.

    En revanche :

        liste = aucun

    ou :

        bibliothèque = aucun

    restent parfaitement valides.
    """

    normalized_users = (
        df["users"]
        .str.lower()
        .str.strip()
    )

    mask = ~normalized_users.isin(
        INVALID_USERS
    )

    filtered_df = df[
        mask
    ].copy()

    removed = len(df) - len(
        filtered_df
    )

    print(
        f"✓ Utilisateurs invalides supprimés : "
        f"{removed:,}"
    )

    return filtered_df


# =============================================================================
# CREATION DE LA RESSOURCE
# =============================================================================

def create_resource(df):
    """
    Crée un identifiant unique représentant
    une ressource SharePoint.

    Une ressource est définie par :

        Site
        Sous-site
        Bibliothèque
        Liste

    Exemple :

        Altrad Services France
        Cash
        aucun
        aucun

    devient :

        Altrad Services France |
        Cash |
        aucun |
        aucun
    """

    df = df.copy()

    df["resource"] = (
        df["site"]
        + " | "
        + df["sous-site"]
        + " | "
        + df["bibliothèque"]
        + " | "
        + df["liste"]
    )

    return df


# =============================================================================
# SUPPRESSION DES DOUBLONS
# =============================================================================

def remove_duplicates(df):
    """
    Supprime les doublons utilisateur / ressource.

    Si :

        adam -> Cash

    apparaît plusieurs fois, le modèle considère
    qu'il s'agit d'une seule interaction.
    """

    before = len(df)

    df = df.drop_duplicates(
        subset=[
            "users",
            "resource",
        ]
    ).copy()

    removed = before - len(df)

    print(
        f"✓ Doublons utilisateur/ressource supprimés : "
        f"{removed:,}"
    )

    return df


# =============================================================================
# CREATION DES INTERACTIONS
# =============================================================================

def create_interactions(df):
    """
    Conserve uniquement les informations
    nécessaires au système de recommandation.

    Une interaction signifie :

        utilisateur X
        utilise / possède
        ressource Y
    """

    interactions = df[
        [
            "users",
            "resource",
        ]
    ].copy()

    interactions = interactions.rename(
        columns={
            "users": "user"
        }
    )

    return interactions


# =============================================================================
# STATISTIQUES
# =============================================================================

def create_statistics(
    df,
    interactions
):
    """
    Calcule les statistiques principales
    du dataset préparé.
    """

    statistics = {
        "rows_original": len(df),

        "users": interactions[
            "user"
        ].nunique(),

        "resources": interactions[
            "resource"
        ].nunique(),

        "interactions": len(
            interactions
        ),

        "sites": df[
            "site"
        ].nunique(),

        "subsites": df[
            "sous-site"
        ].nunique(),

        "libraries": df[
            "bibliothèque"
        ].nunique(),

        "lists": df[
            "liste"
        ].nunique(),
    }

    return statistics


# =============================================================================
# SAUVEGARDE
# =============================================================================

def save_data(
    df,
    interactions
):
    """
    Sauvegarde les données préparées.

    Parquet est utilisé car il est plus efficace
    que CSV pour les traitements ML.
    """

    # Dataset complet préparé.
    df.to_parquet(
        PROCESSED_PATH
        / "clean_data.parquet",
        index=False
    )

    # Interactions utilisateur/ressource.
    interactions.to_parquet(
        PROCESSED_PATH
        / "interactions.parquet",
        index=False
    )

    print(
        "✓ Données sauvegardées dans processed/"
    )


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def run_preprocessing():

    print()
    print("=" * 70)
    print(
        "SHAREPOINT RECOMMENDER - PREPROCESSING"
    )
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 1. Chargement
    # -------------------------------------------------------------------------

    df = load_data()

    # -------------------------------------------------------------------------
    # 2. Validation
    # -------------------------------------------------------------------------

    validate_data(
        df
    )

    # -------------------------------------------------------------------------
    # 3. Normalisation technique
    # -------------------------------------------------------------------------

    df = normalize_columns(
        df
    )

    # -------------------------------------------------------------------------
    # 4. Suppression des faux utilisateurs
    # -------------------------------------------------------------------------

    df = filter_valid_users(
        df
    )

    # -------------------------------------------------------------------------
    # 5. Création de la ressource
    # -------------------------------------------------------------------------

    df = create_resource(
        df
    )

    # -------------------------------------------------------------------------
    # 6. Suppression des doublons
    # -------------------------------------------------------------------------

    df = remove_duplicates(
        df
    )

    # -------------------------------------------------------------------------
    # 7. Création des interactions
    # -------------------------------------------------------------------------

    interactions = create_interactions(
        df
    )

    # -------------------------------------------------------------------------
    # 8. Statistiques
    # -------------------------------------------------------------------------

    statistics = create_statistics(
        df,
        interactions
    )

    # -------------------------------------------------------------------------
    # 9. Sauvegarde
    # -------------------------------------------------------------------------

    save_data(
        df,
        interactions
    )

    # -------------------------------------------------------------------------
    # 10. Résumé
    # -------------------------------------------------------------------------

    print()
    print("=" * 70)
    print("STATISTIQUES")
    print("=" * 70)

    print(
        f"Utilisateurs       : "
        f"{statistics['users']:,}"
    )

    print(
        f"Ressources         : "
        f"{statistics['resources']:,}"
    )

    print(
        f"Interactions       : "
        f"{statistics['interactions']:,}"
    )

    print(
        f"Sites              : "
        f"{statistics['sites']:,}"
    )

    print(
        f"Sous-sites         : "
        f"{statistics['subsites']:,}"
    )

    print(
        f"Bibliothèques      : "
        f"{statistics['libraries']:,}"
    )

    print(
        f"Listes             : "
        f"{statistics['lists']:,}"
    )

    print()
    print(
        "Preprocessing ML terminé."
    )


# =============================================================================
# EXECUTION
# =============================================================================

if __name__ == "__main__":

    run_preprocessing()