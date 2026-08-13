"""
===============================================================================
                    SHAREPOINT RECOMMENDER V2
                         SIMILARITY ENGINE
===============================================================================

Recherche des utilisateurs similaires.

Algorithme :

    K-Nearest Neighbors
    Distance Cosine

Chaque utilisateur est représenté par son profil :

        utilisateur × ressources

Exemple :

        Fabien
           ↓
        ressources auxquelles il a accès
           ↓
        comparaison avec les autres utilisateurs
           ↓
        utilisateurs les plus proches

===============================================================================
"""

from pathlib import Path

import pandas as pd

from sklearn.neighbors import NearestNeighbors


# =============================================================================
# CONFIGURATION
# =============================================================================

PROCESSED_PATH = Path(
    "processed"
)

INTERACTIONS_FILE = (
    PROCESSED_PATH
    / "interactions.parquet"
)

OUTPUT_FILE = (
    PROCESSED_PATH
    / "user_neighbors.parquet"
)


# =============================================================================
# CHARGEMENT
# =============================================================================

def load_interactions():

    if not INTERACTIONS_FILE.exists():

        raise FileNotFoundError(
            """
interactions.parquet introuvable.

Lance d'abord :

python preprocessing.py
"""
        )

    return pd.read_parquet(
        INTERACTIONS_FILE
    )


# =============================================================================
# MATRICE UTILISATEUR / RESSOURCE
# =============================================================================

def create_matrix(
    interactions
):
    """
    Crée :

        lignes    = utilisateurs
        colonnes  = ressources
        valeurs   = 0 ou 1
    """

    matrix = pd.crosstab(
        interactions["user"],
        interactions["resource"]
    )

    matrix = (
        matrix > 0
    ).astype("int8")

    return matrix


# =============================================================================
# KNN
# =============================================================================

def calculate_neighbors(
    matrix,
    n_neighbors=10
):
    """
    Calcule les voisins de chaque utilisateur.

    La distance Cosine permet de comparer
    les profils d'accès SharePoint.
    """

    number_of_users = len(
        matrix
    )

    if number_of_users < 2:

        return pd.DataFrame(
            columns=[
                "user",
                "similar_user",
                "similarity"
            ]
        )

    # +1 car le premier voisin est
    # l'utilisateur lui-même.
    k = min(
        n_neighbors + 1,
        number_of_users
    )

    model = NearestNeighbors(
        n_neighbors=k,
        metric="cosine",
        algorithm="brute"
    )

    model.fit(
        matrix.values
    )

    distances, indices = (
        model.kneighbors(
            matrix.values
        )
    )

    users = matrix.index.tolist()

    results = []

    for i, user in enumerate(users):

        for position in range(
            1,
            len(indices[i])
        ):

            neighbor_index = (
                indices[i][position]
            )

            similar_user = (
                users[neighbor_index]
            )

            distance = (
                distances[i][position]
            )

            similarity = max(
                0,
                1 - distance
            )

            results.append(
                {
                    "user": user,

                    "similar_user":
                        similar_user,

                    "similarity":
                        round(
                            float(
                                similarity
                            ),
                            6
                        )
                }
            )

    return pd.DataFrame(
        results
    )


# =============================================================================
# PIPELINE
# =============================================================================

def run_similarity():

    print()
    print("=" * 70)
    print(
        "CALCUL DES UTILISATEURS SIMILAIRES"
    )
    print("=" * 70)

    interactions = (
        load_interactions()
    )

    print(
        f"Interactions : "
        f"{len(interactions):,}"
    )

    matrix = create_matrix(
        interactions
    )

    print(
        f"Utilisateurs : "
        f"{matrix.shape[0]:,}"
    )

    print(
        f"Ressources : "
        f"{matrix.shape[1]:,}"
    )

    neighbors = calculate_neighbors(
        matrix,
        n_neighbors=10
    )

    neighbors.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"Voisins calculés : "
        f"{len(neighbors):,}"
    )

    print(
        f"✓ Fichier : "
        f"{OUTPUT_FILE}"
    )


# =============================================================================
# EXECUTION
# =============================================================================

if __name__ == "__main__":

    run_similarity()