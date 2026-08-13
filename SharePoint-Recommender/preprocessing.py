"""
============================================================
SHAREPOINT RECOMMENDER - PREPROCESSING V3
============================================================

Préparation des données pour le système de recommandation.

Entrée :
    CSV nettoyé

Sorties :
    processed/interactions.parquet
    processed/resources.parquet

Structure interactions :
    user
    resource

Structure resources :
    resource
    site
    sous-site
    bibliothèque
    liste
"""

from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

PROCESSED_DIR = PROJECT_DIR / "processed"

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FICHIER CSV
# ============================================================

# IMPORTANT :
# Modifie uniquement cette ligne si ton CSV porte
# un autre nom ou se trouve ailleurs.

CSV_FILE = PROJECT_DIR / "Data" / "BD_Sharepoint_Clean.csv"


# ============================================================
# FICHIERS DE SORTIE
# ============================================================

INTERACTIONS_FILE = (
    PROCESSED_DIR / "interactions.parquet"
)

RESOURCES_FILE = (
    PROCESSED_DIR / "resources.parquet"
)


# ============================================================
# UTILISATEURS INVALIDES
# ============================================================

INVALID_USERS = {
    "",
    "aucun",
    "none",
    "null",
    "nan",
    "pas de droit",
    "inconnu",
    "unknown"
}


# ============================================================
# CHARGEMENT DU CSV
# ============================================================

def load_csv():

    if not CSV_FILE.exists():

        raise FileNotFoundError(
            f"\nCSV introuvable : {CSV_FILE}\n"
            "\nModifie CSV_FILE dans preprocessing.py."
        )

    df = pd.read_csv(
        CSV_FILE,
        dtype=str
    )

    # Suppression des espaces inutiles
    df.columns = (
        df.columns
        .str.strip()
    )

    for column in df.columns:

        df[column] = (
            df[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    return df


# ============================================================
# VERIFICATION DE LA STRUCTURE
# ============================================================

def validate_columns(df):

    required_columns = {
        "site",
        "sous-site",
        "liste",
        "bibliothèque",
        "users"
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            "Colonnes manquantes : "
            + ", ".join(sorted(missing))
        )

    print("✓ Structure du CSV valide")


# ============================================================
# NETTOYAGE DES UTILISATEURS
# ============================================================

def clean_users(df):

    initial_count = len(df)

    # Normalisation
    df["users"] = (
        df["users"]
        .str.strip()
        .str.lower()
    )

    # Suppression des utilisateurs invalides
    df = df[
        ~df["users"].isin(
            INVALID_USERS
        )
    ].copy()

    removed = (
        initial_count - len(df)
    )

    print(
        f"✓ Utilisateurs invalides supprimés : "
        f"{removed}"
    )

    return df


# ============================================================
# CREATION DE LA RESSOURCE
# ============================================================

def create_resource_column(df):

    """
    Une ressource est définie par :

        site
        sous-site
        bibliothèque
        liste

    Exemple :

        Altrad Services France |
        Direction financière |
        aucun |
        aucun
    """

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


# ============================================================
# CREATION DE resources.parquet
# ============================================================

def create_resources(df):

    resources = (
        df[
            [
                "resource",
                "site",
                "sous-site",
                "bibliothèque",
                "liste"
            ]
        ]
        .drop_duplicates(
            subset=["resource"]
        )
        .reset_index(drop=True)
    )

    resources.to_parquet(
        RESOURCES_FILE,
        index=False
    )

    return resources


# ============================================================
# CREATION DE interactions.parquet
# ============================================================

def create_interactions(df):

    interactions = (
        df[
            [
                "users",
                "resource"
            ]
        ]
        .rename(
            columns={
                "users": "user"
            }
        )
        .drop_duplicates(
            subset=[
                "user",
                "resource"
            ]
        )
        .reset_index(drop=True)
    )

    interactions.to_parquet(
        INTERACTIONS_FILE,
        index=False
    )

    return interactions


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

def main():

    print()
    print("=" * 70)
    print("SHAREPOINT RECOMMENDER - PREPROCESSING V3")
    print("=" * 70)

    # --------------------------------------------------------
    # Chargement
    # --------------------------------------------------------

    df = load_csv()

    print(
        f"CSV chargé : {len(df):,} lignes"
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validate_columns(df)

    # --------------------------------------------------------
    # Nettoyage utilisateurs
    # --------------------------------------------------------

    df = clean_users(df)

    # --------------------------------------------------------
    # Création ressource
    # --------------------------------------------------------

    df = create_resource_column(df)

    # --------------------------------------------------------
    # Interactions
    # --------------------------------------------------------

    interactions = create_interactions(df)

    # --------------------------------------------------------
    # Ressources
    # --------------------------------------------------------

    resources = create_resources(df)

    # --------------------------------------------------------
    # Statistiques
    # --------------------------------------------------------

    print(
        f"✓ Doublons utilisateur/ressource supprimés : "
        f"{len(df) - len(interactions)}"
    )

    print(
        f"✓ Interactions sauvegardées : "
        f"{INTERACTIONS_FILE}"
    )

    print(
        f"✓ Ressources sauvegardées : "
        f"{RESOURCES_FILE}"
    )

    print()
    print("=" * 70)
    print("STATISTIQUES")
    print("=" * 70)

    print(
        f"Utilisateurs       : "
        f"{interactions['user'].nunique():,}"
    )

    print(
        f"Ressources         : "
        f"{resources['resource'].nunique():,}"
    )

    print(
        f"Interactions       : "
        f"{len(interactions):,}"
    )

    print(
        f"Sites              : "
        f"{resources['site'].nunique():,}"
    )

    print(
        f"Sous-sites         : "
        f"{resources['sous-site'].nunique():,}"
    )

    print(
        f"Bibliothèques      : "
        f"{resources['bibliothèque'].nunique():,}"
    )

    print(
        f"Listes             : "
        f"{resources['liste'].nunique():,}"
    )

    print()
    print("Preprocessing ML terminé.")
    print("=" * 70)


if __name__ == "__main__":
    main()