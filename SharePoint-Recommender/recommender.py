"""
============================================================
SHAREPOINT RECOMMENDER - RECOMMENDER V3
============================================================

Moteur de recommandation hybride.

Combine :

    1. Filtrage collaboratif
    2. Popularité
    3. Contexte SharePoint

Fichiers utilisés :

    processed/interactions.parquet
    processed/resources.parquet
    processed/user_neighbors.parquet
    processed/resource_popularity.parquet

Sortie :

    Top N recommandations pour un utilisateur.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
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
# POIDS DU MODELE
# ============================================================

COLLABORATIVE_WEIGHT = 0.60
POPULARITY_WEIGHT = 0.25
CONTEXT_WEIGHT = 0.15


# ============================================================
# CHARGEMENT DES DONNEES
# ============================================================

def load_data():

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

    return (
        interactions,
        resources,
        neighbors,
        popularity
    )


# ============================================================
# SCORE COLLABORATIF
# ============================================================

def collaborative_scores(
    user,
    interactions,
    neighbors
):
    """
    Calcule le score collaboratif.

    Principe :

    Les utilisateurs similaires à l'utilisateur cible
    servent à identifier les ressources intéressantes.

    Plus un voisin est similaire, plus son interaction
    avec une ressource contribue au score.
    """

    # --------------------------------------------------------
    # Voisins de l'utilisateur
    # --------------------------------------------------------

    user_neighbors = neighbors[
        neighbors["user"] == user
    ].copy()

    if user_neighbors.empty:

        return pd.DataFrame(
            columns=[
                "resource",
                "collaborative_score",
                "support_users"
            ]
        )

    # --------------------------------------------------------
    # Jointure avec les interactions
    # --------------------------------------------------------

    neighbor_interactions = interactions.merge(
        user_neighbors[
            [
                "similar_user",
                "similarity"
            ]
        ],
        left_on="user",
        right_on="similar_user",
        how="inner"
    )

    if neighbor_interactions.empty:

        return pd.DataFrame(
            columns=[
                "resource",
                "collaborative_score",
                "support_users"
            ]
        )

    # --------------------------------------------------------
    # Score collaboratif
    # --------------------------------------------------------

    scores = (
        neighbor_interactions
        .groupby("resource")
        .agg(
            collaborative_score=(
                "similarity",
                "sum"
            ),
            support_users=(
                "similar_user",
                "nunique"
            )
        )
        .reset_index()
    )

    # Normalisation
    max_score = scores[
        "collaborative_score"
    ].max()

    if max_score > 0:

        scores[
            "collaborative_score"
        ] /= max_score

    return scores


# ============================================================
# SCORE DE POPULARITE
# ============================================================

def popularity_scores(
    popularity
):

    scores = popularity[
        [
            "resource",
            "popularity_score",
            "support_users"
        ]
    ].copy()

    return scores


# ============================================================
# SCORE CONTEXTUEL
# ============================================================

def calculate_context_scores(
    user,
    interactions,
    resources
):
    """
    Calcule la proximité contextuelle.

    On regarde les ressources déjà utilisées par l'utilisateur
    puis on mesure combien de leurs caractéristiques sont
    partagées avec une ressource candidate.

    Caractéristiques :

        site
        sous-site
        bibliothèque
        liste
    """

    # --------------------------------------------------------
    # Ressources déjà utilisées
    # --------------------------------------------------------

    user_resources = interactions[
        interactions["user"] == user
    ]["resource"].tolist()

    if not user_resources:

        return pd.DataFrame(
            columns=[
                "resource",
                "context_score"
            ]
        )

    used_resources = resources[
        resources["resource"].isin(
            user_resources
        )
    ]

    if used_resources.empty:

        return pd.DataFrame(
            columns=[
                "resource",
                "context_score"
            ]
        )

    # --------------------------------------------------------
    # Valeurs de contexte utilisées par l'utilisateur
    # --------------------------------------------------------

    user_sites = set(
        used_resources["site"]
    )

    user_subsites = set(
        used_resources["sous-site"]
    )

    user_libraries = set(
        used_resources["bibliothèque"]
    )

    user_lists = set(
        used_resources["liste"]
    )

    # --------------------------------------------------------
    # Calcul pour chaque ressource
    # --------------------------------------------------------

    results = []

    for _, row in resources.iterrows():

        score = 0.0

        if row["site"] in user_sites:
            score += 0.40

        if row["sous-site"] in user_subsites:
            score += 0.30

        if row["bibliothèque"] in user_libraries:
            score += 0.20

        if row["liste"] in user_lists:
            score += 0.10

        results.append(
            {
                "resource": row["resource"],
                "context_score": score
            }
        )

    return pd.DataFrame(results)


# ============================================================
# RECOMMANDATION
# ============================================================

def recommend(
    user,
    interactions,
    resources,
    neighbors,
    popularity,
    top_n=10
):

    # ========================================================
    # Ressources déjà utilisées
    # ========================================================

    used_resources = set(
        interactions[
            interactions["user"] == user
        ]["resource"]
    )

    # ========================================================
    # Score collaboratif
    # ========================================================

    collaborative = collaborative_scores(
        user,
        interactions,
        neighbors
    )

    # ========================================================
    # Score popularité
    # ========================================================

    popular = popularity_scores(
        popularity
    )

    # ========================================================
    # Score contexte
    # ========================================================

    context = calculate_context_scores(
        user,
        interactions,
        resources
    )

    # ========================================================
    # Fusion des scores
    # ========================================================

    recommendations = resources.copy()

    recommendations = recommendations.merge(
        collaborative[
            [
                "resource",
                "collaborative_score",
                "support_users"
            ]
        ],
        on="resource",
        how="left"
    )

    recommendations = recommendations.merge(
        popular[
            [
                "resource",
                "popularity_score"
            ]
        ],
        on="resource",
        how="left"
    )

    recommendations = recommendations.merge(
        context,
        on="resource",
        how="left"
    )

    # ========================================================
    # Valeurs manquantes
    # ========================================================

    recommendations[
        "collaborative_score"
    ] = recommendations[
        "collaborative_score"
    ].fillna(0)

    recommendations[
        "popularity_score"
    ] = recommendations[
        "popularity_score"
    ].fillna(0)

    recommendations[
        "context_score"
    ] = recommendations[
        "context_score"
    ].fillna(0)

    recommendations[
        "support_users"
    ] = recommendations[
        "support_users"
    ].fillna(0)

    # ========================================================
    # SCORE FINAL
    # ========================================================

    recommendations[
        "score"
    ] = (

        COLLABORATIVE_WEIGHT
        * recommendations[
            "collaborative_score"
        ]

        +

        POPULARITY_WEIGHT
        * recommendations[
            "popularity_score"
        ]

        +

        CONTEXT_WEIGHT
        * recommendations[
            "context_score"
        ]
    )

    # ========================================================
    # EXCLUSION DES RESSOURCES DEJA UTILISEES
    # ========================================================

    recommendations = recommendations[
        ~recommendations[
            "resource"
        ].isin(used_resources)
    ]

    # ========================================================
    # TRI
    # ========================================================

    recommendations = (
        recommendations
        .sort_values(
            "score",
            ascending=False
        )
        .head(top_n)
        .copy()
    )

    # ========================================================
    # POURCENTAGE
    # ========================================================

    recommendations[
        "score_percent"
    ] = (
        recommendations["score"]
        * 100
    ).round(2)

    # ========================================================
    # TYPE DE RECOMMANDATION
    # ========================================================

    def recommendation_type(row):

        if row["collaborative_score"] > 0:

            return "Collaborative + contexte"

        if row["context_score"] > 0:

            return "Contexte"

        return "Popularité"

    recommendations[
        "recommendation_type"
    ] = recommendations.apply(
        recommendation_type,
        axis=1
    )

    # ========================================================
    # COLONNES FINALES
    # ========================================================

    columns = [
        "resource",
        "site",
        "sous-site",
        "bibliothèque",
        "liste",
        "score_percent",
        "support_users",
        "recommendation_type"
    ]

    return recommendations[
        columns
    ]


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("SHAREPOINT RECOMMENDER V3")
    print("=" * 70)

    # --------------------------------------------------------
    # Chargement
    # --------------------------------------------------------

    (
        interactions,
        resources,
        neighbors,
        popularity
    ) = load_data()

    # --------------------------------------------------------
    # Utilisateur
    # --------------------------------------------------------

    user = input(
        "\nUtilisateur : "
    ).strip().lower()

    # --------------------------------------------------------
    # Vérification
    # --------------------------------------------------------

    available_users = set(
        interactions["user"]
    )

    if user not in available_users:

        print(
            f"\nUtilisateur '{user}' introuvable."
        )

        print(
            "\nExemples :"
        )

        print(
            ", ".join(
                sorted(
                    available_users
                )[:20]
            )
        )

        raise SystemExit

    # --------------------------------------------------------
    # Recommandations
    # --------------------------------------------------------

    recommendations = recommend(
        user=user,
        interactions=interactions,
        resources=resources,
        neighbors=neighbors,
        popularity=popularity,
        top_n=10
    )

    # --------------------------------------------------------
    # Affichage
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RECOMMANDATIONS")
    print("=" * 70)

    if recommendations.empty:

        print(
            "\nAucune recommandation disponible."
        )

    else:

        print(
            recommendations.to_string(
                index=False
            )
        )

    print()
    print("=" * 70)