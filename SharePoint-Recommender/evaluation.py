"""
===============================================================================
                    SHAREPOINT RECOMMENDER - EVALUATION
===============================================================================

Evaluation du moteur de recommandation.

Métriques :

    Precision@K
    Recall@K
    Hit Rate@K

Principe du test :

    1. On retire temporairement certaines ressources
       utilisées par l'utilisateur.

    2. Le moteur produit des recommandations.

    3. On regarde si les ressources retirées
       sont retrouvées.

===============================================================================
"""

from pathlib import Path

import pandas as pd

from recommender import recommend_resources


PROCESSED_PATH = Path("processed")

USER_RESOURCE_FILE = (
    PROCESSED_PATH / "user_resource.parquet"
)

COSINE_FILE = (
    PROCESSED_PATH / "cosine_similarity.parquet"
)


# =============================================================================
# CHARGEMENT
# =============================================================================

def load_matrices():

    matrix = pd.read_parquet(
        USER_RESOURCE_FILE
    )

    similarity = pd.read_parquet(
        COSINE_FILE
    )

    return matrix, similarity


# =============================================================================
# PRECISION @ K
# =============================================================================

def precision_at_k(
    recommended: list,
    relevant: set,
    k: int,
) -> float:
    """
    Calcule Precision@K.

    Precision@K =
        nombre de recommandations pertinentes
        -------------------------------------
        K
    """

    recommended = recommended[:k]

    if not recommended:
        return 0.0

    hits = sum(
        item in relevant
        for item in recommended
    )

    return hits / len(recommended)


# =============================================================================
# RECALL @ K
# =============================================================================

def recall_at_k(
    recommended: list,
    relevant: set,
    k: int,
) -> float:
    """
    Calcule Recall@K.
    """

    if not relevant:
        return 0.0

    recommended = recommended[:k]

    hits = sum(
        item in relevant
        for item in recommended
    )

    return hits / len(relevant)


# =============================================================================
# HIT RATE @ K
# =============================================================================

def hit_rate_at_k(
    recommended: list,
    relevant: set,
    k: int,
) -> float:
    """
    Retourne 1 si au moins une recommandation
    est pertinente.
    """

    recommended = recommended[:k]

    return float(
        any(
            item in relevant
            for item in recommended
        )
    )


# =============================================================================
# EVALUATION D'UN UTILISATEUR
# =============================================================================

def evaluate_user(
    user: str,
    matrix: pd.DataFrame,
    similarity: pd.DataFrame,
    k: int = 10,
) -> dict:

    resources = (
        matrix.loc[user]
        [matrix.loc[user] > 0]
        .index
        .tolist()
    )

    # Il faut suffisamment de ressources
    # pour créer un test.
    if len(resources) < 2:

        return {
            "user": user,
            "precision": 0.0,
            "recall": 0.0,
            "hit_rate": 0.0,
        }

    # Une ressource sert de test.
    hidden_resource = resources[-1]

    # Copie de la matrice.
    test_matrix = matrix.copy()

    test_matrix.loc[
        user,
        hidden_resource,
    ] = 0

    # Génération des recommandations.
    recommendations = recommend_resources(
        user=user,
        similarity_matrix=similarity,
        user_resource_matrix=test_matrix,
        n_users=10,
        n_recommendations=k,
    )

    if recommendations.empty:

        recommended = []

    else:

        recommended = (
            recommendations[
                "resource"
            ]
            .tolist()
        )

    relevant = {
        hidden_resource
    }

    return {
        "user": user,
        "precision": precision_at_k(
            recommended,
            relevant,
            k,
        ),
        "recall": recall_at_k(
            recommended,
            relevant,
            k,
        ),
        "hit_rate": hit_rate_at_k(
            recommended,
            relevant,
            k,
        ),
    }


# =============================================================================
# EVALUATION GLOBALE
# =============================================================================

def evaluate_model(
    max_users: int = 100,
    k: int = 10,
) -> pd.DataFrame:
    """
    Évalue le moteur sur plusieurs utilisateurs.

    Parameters
    ----------
    max_users : int
        Nombre maximal d'utilisateurs évalués.

    k : int
        Nombre de recommandations.

    Returns
    -------
    pd.DataFrame
    """

    matrix, similarity = load_matrices()

    # On limite le nombre d'utilisateurs
    # pour garder l'évaluation rapide.
    users = matrix.index[:max_users]

    results = []

    for user in users:

        result = evaluate_user(
            user=user,
            matrix=matrix,
            similarity=similarity,
            k=k,
        )

        results.append(result)

    return pd.DataFrame(results)


# =============================================================================
# RESUME
# =============================================================================

def summarize_evaluation(
    results: pd.DataFrame,
) -> dict:
    """
    Calcule les moyennes globales.
    """

    return {
        "precision_at_k": round(
            results["precision"].mean(),
            4,
        ),
        "recall_at_k": round(
            results["recall"].mean(),
            4,
        ),
        "hit_rate_at_k": round(
            results["hit_rate"].mean(),
            4,
        ),
    }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":

    results = evaluate_model(
        max_users=100,
        k=10,
    )

    summary = summarize_evaluation(
        results
    )

    print(
        "\n======================================"
    )

    print(
        "EVALUATION DU MODELE"
    )

    print(
        "======================================"
    )

    print(
        f"\nPrecision@10 : "
        f"{summary['precision_at_k']}"
    )

    print(
        f"Recall@10    : "
        f"{summary['recall_at_k']}"
    )

    print(
        f"Hit Rate@10  : "
        f"{summary['hit_rate_at_k']}"
    )