"""
============================================================
RECOMMENDER.PY - VERSION 3
============================================================

SYSTÈME DE RECOMMANDATION HYBRIDE SHAREPOINT

Le moteur combine :

    1. Similarité entre utilisateurs
    2. Popularité des ressources
    3. Contexte SharePoint

Score final :

    60 % -> collaboratif
    25 % -> popularité
    15 % -> contexte
"""

from pathlib import Path

import pandas as pd

from context import calculate_context_score
from popularity import calculate_popularity


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
PROCESSED_DIR = PROJECT_DIR / "processed"

INTERACTIONS_FILE = (
    PROCESSED_DIR / "interactions.parquet"
)

NEIGHBORS_FILE = (
    PROCESSED_DIR / "user_neighbors.parquet"
)


# ============================================================
# POIDS DU MODÈLE
# ============================================================

COLLAB_WEIGHT = 0.60
POPULARITY_WEIGHT = 0.25
CONTEXT_WEIGHT = 0.15


# ============================================================
# CHARGEMENT DES DONNÉES
# ============================================================

def load_data():

    interactions = pd.read_parquet(
        INTERACTIONS_FILE
    )

    neighbors = pd.read_parquet(
        NEIGHBORS_FILE
    )

    return interactions, neighbors


# ============================================================
# RECOMMANDATION COLLABORATIVE
# ============================================================

def collaborative_recommendations(
    user,
    interactions,
    neighbors
):
    """
    Génère les recommandations à partir des utilisateurs
    similaires.

    Le fichier neighbors contient :

        user
        similar_user
        similarity
    """

    # --------------------------------------------------------
    # Ressources déjà utilisées
    # --------------------------------------------------------

    user_resources = set(
        interactions.loc[
            interactions["users"] == user,
            "resource"
        ]
    )

    # --------------------------------------------------------
    # Utilisateurs similaires
    # --------------------------------------------------------

    user_neighbors = neighbors[
        neighbors["user"] == user
    ]

    if user_neighbors.empty:

        return pd.DataFrame(
            columns=[
                "resource",
                "collaborative_score",
                "support_users"
            ]
        )

    scores = {}

    # --------------------------------------------------------
    # Parcours des utilisateurs similaires
    # --------------------------------------------------------

    for _, neighbor in user_neighbors.iterrows():

        similar_user = neighbor["similar_user"]

        similarity = float(
            neighbor["similarity"]
        )

        # Ressources du voisin
        similar_resources = interactions.loc[
            interactions["users"] == similar_user,
            "resource"
        ]

        # ----------------------------------------------------
        # Pondération par similarité
        # ----------------------------------------------------

        for resource in similar_resources:

            # Ne jamais recommander une ressource
            # déjà utilisée
            if resource in user_resources:
                continue

            scores[resource] = (
                scores.get(resource, 0.0)
                + similarity
            )

    # --------------------------------------------------------
    # Aucun résultat
    # --------------------------------------------------------

    if not scores:

        return pd.DataFrame(
            columns=[
                "resource",
                "collaborative_score",
                "support_users"
            ]
        )

    # --------------------------------------------------------
    # Conversion en DataFrame
    # --------------------------------------------------------

    result = pd.DataFrame(
        list(scores.items()),
        columns=[
            "resource",
            "collaborative_score"
        ]
    )

    # --------------------------------------------------------
    # Normalisation
    # --------------------------------------------------------

    max_score = result[
        "collaborative_score"
    ].max()

    if max_score > 0:

        result["collaborative_score"] = (
            result["collaborative_score"]
            / max_score
        )

    # --------------------------------------------------------
    # Support utilisateur
    # --------------------------------------------------------

    support = (
        interactions
        .groupby("resource")["users"]
        .nunique()
        .reset_index(
            name="support_users"
        )
    )

    result = result.merge(
        support,
        on="resource",
        how="left"
    )

    return result


# ============================================================
# RECOMMANDATION HYBRIDE
# ============================================================

def recommend(
    user,
    top_n=10
):
    """
    Génère les recommandations hybrides.
    """

    interactions, neighbors = load_data()

    # --------------------------------------------------------
    # Vérification utilisateur
    # --------------------------------------------------------

    if user not in interactions["users"].values:

        raise ValueError(
            f"Utilisateur inconnu : {user}"
        )

    # --------------------------------------------------------
    # Liste des ressources
    # --------------------------------------------------------

    resources = (
        interactions[
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
    )

    # --------------------------------------------------------
    # SCORE COLLABORATIF
    # --------------------------------------------------------

    collaborative = (
        collaborative_recommendations(
            user,
            interactions,
            neighbors
        )
    )

    # --------------------------------------------------------
    # Si aucun résultat collaboratif
    # --------------------------------------------------------

    if collaborative.empty:

        return popularity_fallback(
            user,
            interactions,
            top_n
        )

    # --------------------------------------------------------
    # SCORE POPULARITÉ
    # --------------------------------------------------------

    popularity = calculate_popularity(
        interactions
    )

    # --------------------------------------------------------
    # Fusion collaborative + ressources
    # --------------------------------------------------------

    result = resources.merge(
        collaborative[
            [
                "resource",
                "collaborative_score",
                "support_users"
            ]
        ],
        on="resource",
        how="inner"
    )

    # --------------------------------------------------------
    # Popularité
    # --------------------------------------------------------

    result = result.merge(
        popularity[
            [
                "resource",
                "popularity_score"
            ]
        ],
        on="resource",
        how="left"
    )

    # --------------------------------------------------------
    # SCORE CONTEXTUEL
    # --------------------------------------------------------

    result["context_score"] = result.apply(
        lambda row:
        calculate_context_score(
            user,
            row,
            interactions
        ),
        axis=1
    )

    # --------------------------------------------------------
    # Valeurs manquantes
    # --------------------------------------------------------

    result[
        [
            "collaborative_score",
            "popularity_score",
            "context_score"
        ]
    ] = result[
        [
            "collaborative_score",
            "popularity_score",
            "context_score"
        ]
    ].fillna(0)

    # --------------------------------------------------------
    # SCORE FINAL
    # --------------------------------------------------------

    result["final_score"] = (

        COLLAB_WEIGHT
        * result["collaborative_score"]

        +

        POPULARITY_WEIGHT
        * result["popularity_score"]

        +

        CONTEXT_WEIGHT
        * result["context_score"]
    )

    # --------------------------------------------------------
    # POURCENTAGE
    # --------------------------------------------------------

    result["score_percent"] = (
        result["final_score"] * 100
    ).round(2)

    # --------------------------------------------------------
    # TYPE
    # --------------------------------------------------------

    result["recommendation_type"] = "Hybride"

    # --------------------------------------------------------
    # TRI
    # --------------------------------------------------------

    result = result.sort_values(
        "final_score",
        ascending=False
    )

    return result.head(top_n)


# ============================================================
# FALLBACK POPULARITÉ
# ============================================================

def popularity_fallback(
    user,
    interactions,
    top_n=10
):
    """
    Fallback utilisé lorsqu'aucune recommandation
    collaborative n'est disponible.

    On recommande alors les ressources les plus populaires
    que l'utilisateur n'utilise pas encore.
    """

    # Ressources déjà utilisées
    user_resources = set(
        interactions.loc[
            interactions["users"] == user,
            "resource"
        ]
    )

    # Popularité
    popularity = calculate_popularity(
        interactions
    )

    # Ressources
    resources = (
        interactions[
            [
                "resource",
                "site",
                "sous-site",
                "bibliothèque",
                "liste"
            ]
        ]
        .drop_duplicates("resource")
    )

    # Fusion
    result = resources.merge(
        popularity,
        on="resource",
        how="left"
    )

    # Exclusion
    result = result[
        ~result["resource"].isin(
            user_resources
        )
    ]

    # Score
    result["final_score"] = (
        result["popularity_score"]
    )

    result["score_percent"] = (
        result["final_score"] * 100
    ).round(2)

    result["collaborative_score"] = 0.0
    result["context_score"] = 0.0

    result["recommendation_type"] = (
        "Popularité"
    )

    return (
        result
        .sort_values(
            "final_score",
            ascending=False
        )
        .head(top_n)
    )


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("SHAREPOINT RECOMMENDER V3")
    print("=" * 70)

    user = input(
        "\nUtilisateur : "
    ).strip()

    try:

        recommendations = recommend(
            user,
            top_n=10
        )

        print("\n" + "=" * 70)
        print(
            f"RECOMMANDATIONS POUR : {user}"
        )
        print("=" * 70)

        print(
            recommendations[
                [
                    "resource",
                    "site",
                    "sous-site",
                    "bibliothèque",
                    "liste",
                    "collaborative_score",
                    "popularity_score",
                    "context_score",
                    "score_percent",
                    "support_users",
                    "recommendation_type"
                ]
            ].to_string(
                index=False
            )
        )

    except Exception as e:

        print(
            f"\nErreur : {e}"
        )