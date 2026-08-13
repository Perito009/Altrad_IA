"""
===============================================================================
                    SHAREPOINT RECOMMENDER V2
                           EVALUATION
===============================================================================

Evaluation du système de recommandation.

Métriques :

    Precision@K
    Recall@K
    Hit Rate@K
    NDCG@K

Attention :

Cette première version utilise une interaction masquée
par utilisateur.

Pour une évaluation industrielle, on pourra ensuite
mettre en place une séparation temporelle ou un
train/test split plus complet.

===============================================================================
"""

from pathlib import Path

import math
import pandas as pd

from recommender import (
    recommend_resources
)


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

NEIGHBORS_FILE = (
    PROCESSED_PATH
    / "user_neighbors.parquet"
)

CLEAN_DATA_FILE = (
    PROCESSED_PATH
    / "clean_data.parquet"
)


# =============================================================================
# PRECISION@K
# =============================================================================

def precision_at_k(
    recommended,
    relevant,
    k
):
    """
    Nombre de recommandations pertinentes
    parmi les K premières recommandations.
    """

    recommended = recommended[
        :k
    ]

    if not recommended:

        return 0.0

    hits = sum(
        item in relevant
        for item in recommended
    )

    return hits / len(
        recommended
    )


# =============================================================================
# RECALL@K
# =============================================================================

def recall_at_k(
    recommended,
    relevant,
    k
):
    """
    Part des éléments pertinents retrouvés
    dans les K recommandations.
    """

    if not relevant:

        return 0.0

    recommended = recommended[
        :k
    ]

    hits = sum(
        item in relevant
        for item in recommended
    )

    return hits / len(
        relevant
    )


# =============================================================================
# HIT RATE@K
# =============================================================================

def hit_rate_at_k(
    recommended,
    relevant,
    k
):
    """
    Retourne 1 si au moins une ressource
    pertinente est retrouvée.
    """

    recommended = recommended[
        :k
    ]

    return float(
        any(
            item in relevant
            for item in recommended
        )
    )


# =============================================================================
# NDCG@K
# =============================================================================

def ndcg_at_k(
    recommended,
    relevant,
    k
):
    """
    NDCG mesure la qualité du classement.

    Une bonne recommandation placée en haut
    de la liste reçoit plus de poids.
    """

    recommended = recommended[
        :k
    ]

    dcg = 0.0

    for position, item in enumerate(
        recommended,
        start=1
    ):

        if item in relevant:

            dcg += (
                1
                / math.log2(
                    position + 1
                )
            )

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

    return dcg / idcg


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(
    max_users=100,
    k=10
):

    interactions = pd.read_parquet(
        INTERACTIONS_FILE
    )

    neighbors = pd.read_parquet(
        NEIGHBORS_FILE
    )

    clean_data = pd.read_parquet(
        CLEAN_DATA_FILE
    )

    users = (
        interactions[
            "user"
        ]
        .unique()
        .tolist()
    )

    users = users[
        :max_users
    ]

    results = []

    for user in users:

        user_rows = interactions[
            interactions["user"] == user
        ]

        # Il faut au minimum deux ressources
        # pour cacher une interaction.
        if len(user_rows) < 2:

            continue

        # ---------------------------------------------------------------------
        # Ressource cachée
        # ---------------------------------------------------------------------

        hidden_row = (
            user_rows.iloc[-1]
        )

        hidden_resource = (
            hidden_row[
                "resource"
            ]
        )

        # ---------------------------------------------------------------------
        # Dataset d'entraînement temporaire
        # ---------------------------------------------------------------------

        train_interactions = (
            interactions.drop(
                index=hidden_row.name
            )
        )

        # ---------------------------------------------------------------------
        # Recommandations
        # ---------------------------------------------------------------------

        recommendations = (
            recommend_resources(
                user=user,
                interactions=
                    train_interactions,
                neighbors=neighbors,
                clean_data=clean_data,
                n_users=10,
                n_recommendations=k
            )
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

        results.append(
            {
                "user": user,

                "precision_at_k":
                    precision_at_k(
                        recommended,
                        relevant,
                        k
                    ),

                "recall_at_k":
                    recall_at_k(
                        recommended,
                        relevant,
                        k
                    ),

                "hit_rate_at_k":
                    hit_rate_at_k(
                        recommended,
                        relevant,
                        k
                    ),

                "ndcg_at_k":
                    ndcg_at_k(
                        recommended,
                        relevant,
                        k
                    ),
            }
        )

    return pd.DataFrame(
        results
    )


# =============================================================================
# RESUME
# =============================================================================

def summarize(
    results
):

    if results.empty:

        return {
            "precision": 0,
            "recall": 0,
            "hit_rate": 0,
            "ndcg": 0
        }

    return {
        "precision":
            round(
                results[
                    "precision_at_k"
                ].mean(),
                4
            ),

        "recall":
            round(
                results[
                    "recall_at_k"
                ].mean(),
                4
            ),

        "hit_rate":
            round(
                results[
                    "hit_rate_at_k"
                ].mean(),
                4
            ),

        "ndcg":
            round(
                results[
                    "ndcg_at_k"
                ].mean(),
                4
            ),
    }


# =============================================================================
# EXECUTION
# =============================================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print(
        "EVALUATION DU SYSTEME V2"
    )
    print("=" * 70)

    results = evaluate(
        max_users=100,
        k=10
    )

    summary = summarize(
        results
    )

    print()

    print(
        f"Utilisateurs évalués : "
        f"{len(results)}"
    )

    print(
        f"Precision@10 : "
        f"{summary['precision']}"
    )

    print(
        f"Recall@10    : "
        f"{summary['recall']}"
    )

    print(
        f"Hit Rate@10  : "
        f"{summary['hit_rate']}"
    )

    print(
        f"NDCG@10      : "
        f"{summary['ndcg']}"
    )