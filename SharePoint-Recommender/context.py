"""
============================================================
CONTEXT.PY
============================================================

Calcul du score contextuel des ressources SharePoint.

Le contexte est basé sur :

    - site
    - sous-site
    - bibliothèque
    - liste

L'objectif est de déterminer si une ressource recommandée
appartient à un environnement SharePoint proche de celui
déjà utilisé par l'utilisateur.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "processed"

INTERACTIONS_FILE = PROCESSED_DIR / "interactions.parquet"


# ============================================================
# CHARGEMENT
# ============================================================

def load_interactions():
    """
    Charge les interactions utilisateur / ressource.
    """

    if not INTERACTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {INTERACTIONS_FILE}"
        )

    return pd.read_parquet(INTERACTIONS_FILE)


# ============================================================
# SCORE CONTEXTUEL
# ============================================================

def calculate_context_score(user, candidate, interactions):
    """
    Calcule la proximité contextuelle entre :

        user      = utilisateur
        candidate = ressource candidate

    Le score utilise :

        site        : 40 %
        sous-site   : 30 %
        bibliothèque : 20 %
        liste       : 10 %

    Le résultat est compris entre 0 et 1.
    """

    user_data = interactions[
        interactions["users"] == user
    ]

    if user_data.empty:
        return 0.0

    score = 0.0

    # --------------------------------------------------------
    # SITE
    # --------------------------------------------------------

    if candidate["site"] in user_data["site"].values:
        score += 0.40

    # --------------------------------------------------------
    # SOUS-SITE
    # --------------------------------------------------------

    if candidate["sous-site"] in user_data["sous-site"].values:
        score += 0.30

    # --------------------------------------------------------
    # BIBLIOTHÈQUE
    # --------------------------------------------------------

    if candidate["bibliothèque"] in user_data["bibliothèque"].values:
        score += 0.20

    # --------------------------------------------------------
    # LISTE
    # --------------------------------------------------------

    if candidate["liste"] in user_data["liste"].values:
        score += 0.10

    return score


# ============================================================
# TEST DU MODULE
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("CALCUL DU SCORE CONTEXTUEL")
    print("=" * 70)

    df = load_interactions()

    print(f"Interactions : {len(df):,}")
    print(f"Utilisateurs : {df['users'].nunique():,}")

    print("\nModule contextuel chargé")