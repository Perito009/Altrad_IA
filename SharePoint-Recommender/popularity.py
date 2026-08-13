"""
============================================================
POPULARITY.PY
============================================================

Calcul de la popularité des ressources SharePoint.

Source :
    processed/interactions.parquet

Structure réelle :
    user
    resource

La popularité d'une ressource correspond au nombre
d'utilisateurs différents qui utilisent cette ressource.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# CHEMINS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
PROCESSED_DIR = PROJECT_DIR / "processed"

INTERACTIONS_FILE = PROCESSED_DIR / "interactions.parquet"
OUTPUT_FILE = PROCESSED_DIR / "resource_popularity.parquet"


# ============================================================
# CALCUL
# ============================================================

def calculate_popularity(interactions):
    """
    Calcule la popularité de chaque ressource.

    support_users :
        nombre d'utilisateurs différents utilisant
        la ressource.

    popularity_score :
        score normalisé entre 0 et 1.
    """

    popularity = (
        interactions
        .groupby("resource")["user"]
        .nunique()
        .reset_index(name="support_users")
    )

    # Normalisation
    max_support = popularity["support_users"].max()

    if max_support > 0:
        popularity["popularity_score"] = (
            popularity["support_users"] / max_support
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

    # Vérification du fichier
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

    print(
        f"Utilisateurs : "
        f"{interactions['user'].nunique():,}"
    )

    print(
        f"Ressources : "
        f"{interactions['resource'].nunique():,}"
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
        f"\n✓ Fichier créé : {OUTPUT_FILE}"
    )

    print("\nTop 10 des ressources populaires :")
    print("-" * 70)

    print(
        popularity
        .sort_values(
            "support_users",
            ascending=False
        )
        .head(10)
        .to_string(index=False)
    )

    print("\n" + "=" * 70)