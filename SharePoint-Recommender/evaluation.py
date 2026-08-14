"""
============================================================
SHAREPOINT RECOMMENDER - EVALUATION V3
============================================================

Evaluation du système de recommandation.

Métriques :

    Precision@10
    Recall@10
    Hit Rate@10
    NDCG@10

Méthode :

Pour chaque utilisateur :

    1. Une partie de ses interactions est masquée.
    2. Le système produit des recommandations.
    3. Les recommandations sont comparées aux interactions
       réellement présentes dans la partie test.

Cela permet d'évaluer la capacité du modèle à retrouver
des ressources connues mais volontairement masquées.
"""

from pathlib import Path
import sys
import math

import pandas as pd


# ============================================================
# CHEMINS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
PROCESSED_DIR = PROJECT_DIR / "processed"

INTERACTIONS_FILE = (
    PROCESSED_DIR / "interactions.parquet"
)

RESOURCES_FILE = (
    PROCESSED_DIR / "resources.parquet"
)

NEIGHBORS_FILE = (
    PROCESSED_DIR / "user_neighbors.parquet"
)

POPULARITY_FILE = (
    PROCESSED_DIR / "resource_popularity.parquet"
)


# ============================================================
# IMPORT DU RECOMMENDER
# ============================================================

# Permet d'importer recommender.py lorsqu'on exécute
# evaluation.py depuis la racine du projet.

sys.path.insert(
    0,
    str(BASE_DIR)
)

from recommender import recommend


# ============================================================
# CONFIGURATION
# ============================================================

TOP_K = 10

# Nombre maximum d'utilisateurs évalués.
# 100 permet d'avoir une évaluation suffisamment rapide.
MAX_USERS = 100

# Proportion des interactions conservées pour le test.
TEST_RATIO = 0.20

RANDOM_STATE = 42


# ============================================================
# PRECISION@K
# ============================================================

def precision_at_k(
    recommended,
    relevant,
    k=10
):
    """
    Precision@K :

        nombre de recommandations pertinentes
        -------------------------------------
                    K
    """

    recommended = recommended[:k]

    if not recommended:
        return 0.0

    hits = len(
        set(recommended)
        & set(relevant)
    )

    return hits / k


# ============================================================
# RECALL@K
# ============================================================

def recall_at_k(
    recommended,
    relevant,
    k=10
):
    """
    Recall@K :

        nombre de recommandations pertinentes
        -------------------------------------
        nombre total de ressources pertinentes
    """

    if not relevant:
        return 0.0

    recommended = recommended[:k]

    hits = len(
        set(recommended)
        & set(relevant)
    )

    return hits / len(relevant)


# ============================================================
# HIT RATE@K
# ============================================================

def hit_rate_at_k(
    recommended,
    relevant,
    k=10
):
    """
    Retourne 1 si au moins une ressource pertinente
    apparaît dans les K recommandations.
    """

    recommended = recommended[:k]

    return float(
        len(
            set(recommended)
            & set(relevant)
        ) > 0
    )


# ============================================================
# NDCG@K
# ============================================================

def ndcg_at_k(
    recommended,
    relevant,
    k=10
):
    """
    Normalized Discounted Cumulative Gain.

    Cette métrique prend en compte la position
    de la recommandation.

    Une bonne recommandation placée en position 1
    rapporte davantage qu'une bonne recommandation
    placée en position 10.
    """

    recommended = recommended[:k]

    relevant = set(relevant)

    dcg = 0.0

    for index, resource in enumerate(
        recommended
    ):

        if resource in relevant:

            position = index + 1

            dcg += (
                1
                / math.log2(
                    position + 1
                )
            )

    # --------------------------------------------------------
    # IDCG
    # --------------------------------------------------------

    ideal_hits = min(
        len(relevant),
        k
    )

    if ideal_hits == 0:
        return 0.0

    idcg = sum(
        1
        / math.log2(
            position + 1
        )
        for position in range(
            1,
            ideal_hits + 1
        )
    )

    if idcg == 0:
        return 0.0

    return dcg / idcg


# ============================================================
# EVALUATION D'UN UTILISATEUR
# ============================================================

def evaluate_user(
    user,
    train_interactions,
    test_resources,
    resources,
    neighbors,
    popularity
):
    """
    Evalue un utilisateur.

    train_interactions :
        interactions utilisées par le recommender.

    test_resources :
        ressources volontairement cachées
        servant de vérité terrain.
    """

    # --------------------------------------------------------
    # Recommandations
    # --------------------------------------------------------

    recommendations = recommend(
        user=user,
        interactions=train_interactions,
        resources=resources,
        neighbors=neighbors,
        popularity=popularity,
        top_n=TOP_K
    )

    if recommendations.empty:

        return {
            "precision": 0.0,
            "recall": 0.0,
            "hit_rate": 0.0,
            "ndcg": 0.0
        }

    recommended_resources = (
        recommendations[
            "resource"
        ]
        .tolist()
    )

    relevant_resources = list(
        test_resources
    )

    # --------------------------------------------------------
    # Métriques
    # --------------------------------------------------------

    precision = precision_at_k(
        recommended_resources,
        relevant_resources,
        TOP_K
    )

    recall = recall_at_k(
        recommended_resources,
        relevant_resources,
        TOP_K
    )

    hit_rate = hit_rate_at_k(
        recommended_resources,
        relevant_resources,
        TOP_K
    )

    ndcg = ndcg_at_k(
        recommended_resources,
        relevant_resources,
        TOP_K
    )

    return {
        "precision": precision,
        "recall": recall,
        "hit_rate": hit_rate,
        "ndcg": ndcg
    }


# ============================================================
# EVALUATION GLOBALE
# ============================================================

def evaluate():

    print()
    print("=" * 70)
    print("EVALUATION DU SYSTEME DE RECOMMANDATION V3")
    print("=" * 70)

    # --------------------------------------------------------
    # Chargement
    # --------------------------------------------------------

    interactions = pd.read_parquet(
        INTERACTIONS_FILE
    )

    resources = pd.read_parquet(
        RESOURCES_FILE
    )

    neighbors = pd.read_parquet(
        NEIGHBORS_FILE
    )

    popularity = pd.read_parquet(
        POPULARITY_FILE
    )

    print(
        f"\nInteractions : "
        f"{len(interactions):,}"
    )

    print(
        f"Utilisateurs : "
        f"{interactions['user'].nunique():,}"
    )

    print(
        f"Ressources : "
        f"{interactions['resource'].nunique():,}"
    )

    # --------------------------------------------------------
    # Sélection des utilisateurs
    # --------------------------------------------------------

    users = (
        interactions["user"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    # Limitation pour accélérer l'évaluation
    users = users[:MAX_USERS]

    print(
        f"\nUtilisateurs évalués : "
        f"{len(users)}"
    )

    # --------------------------------------------------------
    # Résultats
    # --------------------------------------------------------

    results = []

    # --------------------------------------------------------
    # Evaluation utilisateur par utilisateur
    # --------------------------------------------------------

    for user_index, user in enumerate(
        users,
        start=1
    ):

        user_data = interactions[
            interactions["user"] == user
        ]

        # ----------------------------------------------------
        # Il faut suffisamment d'interactions
        # ----------------------------------------------------

        if len(user_data) < 2:
            continue

        # ----------------------------------------------------
        # Mélange reproductible
        # ----------------------------------------------------

        shuffled = user_data.sample(
            frac=1.0,
            random_state=(
                RANDOM_STATE
                + user_index
            )
        )

        # ----------------------------------------------------
        # Nombre d'interactions de test
        # ----------------------------------------------------

        test_size = max(
            1,
            int(
                len(shuffled)
                * TEST_RATIO
            )
        )

        # ----------------------------------------------------
        # Séparation train / test
        # ----------------------------------------------------

        test_part = shuffled.iloc[
            :test_size
        ]

        train_part = shuffled.iloc[
            test_size:
        ]

        # ----------------------------------------------------
        # Sécurité
        # ----------------------------------------------------

        if train_part.empty:
            continue

        test_resources = set(
            test_part["resource"]
        )

        # ----------------------------------------------------
        # Evaluation
        # ----------------------------------------------------

        metrics = evaluate_user(
            user=user,
            train_interactions=train_part,
            test_resources=test_resources,
            resources=resources,
            neighbors=neighbors,
            popularity=popularity
        )

        results.append(
            metrics
        )

    # ========================================================
    # RESULTATS
    # ========================================================

    if not results:

        print(
            "\nAucun utilisateur n'a pu être évalué."
        )

        return

    results_df = pd.DataFrame(
        results
    )

    precision = (
        results_df["precision"]
        .mean()
    )

    recall = (
        results_df["recall"]
        .mean()
    )

    hit_rate = (
        results_df["hit_rate"]
        .mean()
    )

    ndcg = (
        results_df["ndcg"]
        .mean()
    )

    # ========================================================
    # AFFICHAGE
    # ========================================================

    print()
    print("=" * 70)
    print("RESULTATS")
    print("=" * 70)

    print(
        f"\nUtilisateurs évalués : "
        f"{len(results_df)}"
    )

    print(
        f"Precision@10 : "
        f"{precision:.2f}"
    )

    print(
        f"Recall@10    : "
        f"{recall:.2f}"
    )

    print(
        f"Hit Rate@10  : "
        f"{hit_rate:.2f}"
    )

    print(
        f"NDCG@10      : "
        f"{ndcg:.2f}"
    )

    print()
    print("=" * 70)


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":

    evaluate()