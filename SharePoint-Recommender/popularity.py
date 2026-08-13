"""
============================================================
POPULARITY.PY
============================================================

Calcul de la popularité des ressources SharePoint.

Une ressource est considérée comme populaire lorsqu'elle
est utilisée par un grand nombre d'utilisateurs différents.

Exemple :

    Ressource A -> 20 utilisateurs
    Ressource B -> 10 utilisateurs
    Ressource C -> 5 utilisateurs

La ressource A sera donc plus populaire que B et C.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION DES CHEMINS
# ============================================================

# Dossier contenant les scripts
BASE_DIR = Path(__file__).resolve().parent

# Racine du projet Altrad_IA-main
PROJECT_DIR = BASE_DIR.parent

# Dossier des données générées
PROCESSED_DIR = PROJECT_DIR / "processed"

# Fichier des interactions
INTERACTIONS_FILE = PROCESSED_DIR / "interactions.parquet"

# Fichier de sortie
OUTPUT_FILE = PROCESSED_DIR / "resource_popularity.parquet"


# ============================================================
# CALCUL DE LA POPULARITÉ
# ============================================================

def calculate_popularity(interactions):
    """
    Calcule le nombre d'utilisateurs distincts
    pour chaque ressource.
    """

    popularity = (
        interactions
        .groupby("resource")["users"]
        .nunique()
        .reset_index(name="support_users")
    )

    # --------------------------------------------------------
    # Normalisation entre 0 et 1
    # --------------------------------------------------------

    max_support = popularity["support_users"].max()

    if max_support > 0:

        popularity["popularity_score"] = (
            popularity["support_users"]
            / max_support
        )

    else:

        popularity["popularity_score"] = 0.0

    return popularity


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("CALCUL DE LA POPULARITÉ DES RESSOURCES")
    print("=" * 70)

    # Vérification
    if not INTERACTIONS_FILE.exists():

        raise FileNotFoundError(
            f"Fichier introuvable : {INTERACTIONS_FILE}"
        )

    # Chargement
    interactions = pd.read_parquet(
        INTERACTIONS_FILE
    )

    print(
        f"Interactions : {len(interactions):,}"
    )

    # Calcul
    popularity = calculate_popularity(
        interactions
    )

    # Sauvegarde
    popularity.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"Ressources analysées : "
        f"{len(popularity):,}"
    )

    print(
        f"Fichier créé : {OUTPUT_FILE}"
    )

    print("\nTop 10 des ressources populaires :")

    print(
        popularity
        .sort_values(
            "support_users",
            ascending=False
        )
        .head(10)
        .to_string(index=False)
    )