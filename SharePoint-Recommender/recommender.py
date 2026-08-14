"""
============================================================
SHAREPOINT RECOMMENDER - V3.1
============================================================

Système de recommandation hybride et explicable.

Le moteur combine trois informations :

1. Collaboration
   -> comportement des utilisateurs similaires

2. Contexte SharePoint
   -> site / sous-site / bibliothèque / liste

3. Popularité
   -> nombre d'utilisateurs utilisant la ressource

Le résultat fournit également une explication
pour chaque recommandation.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION PANDAS
# ============================================================

# Évite les warnings liés aux changements futurs
# de comportement de pandas.

pd.set_option(
    "future.no_silent_downcasting",
    True
)


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
# PARAMETRES DU MODELE
# ============================================================

# Poids des différentes composantes.

COLLAB_WEIGHT = 0.70
CONTEXT_WEIGHT = 0.20
POPULARITY_WEIGHT = 0.10


# Nombre maximal d'utilisateurs similaires
# utilisés pour les recommandations.

MAX_NEIGHBORS = 20


# ============================================================
# CHARGEMENT DES DONNEES
# ============================================================

def load_data():
    """
    Charge toutes les données nécessaires au moteur.
    """

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
# SCORE CONTEXTUEL
# ============================================================

def calculate_context_score(
    resource_row,
    user_resources,
    resources
):
    """
    Calcule la proximité contextuelle d'une ressource.

    On compare :

        site
        sous-site
        bibliothèque
        liste

    avec les ressources déjà utilisées
    par l'utilisateur.

    Les niveaux les plus précis ont plus de poids.
    """

    if not user_resources:
        return 0.0

    best_score = 0.0

    # --------------------------------------------------------
    # Ressources déjà utilisées par l'utilisateur
    # --------------------------------------------------------

    user_resource_rows = resources[
        resources["resource"].isin(
            user_resources
        )
    ]

    if user_resource_rows.empty:
        return 0.0

    # --------------------------------------------------------
    # Comparaison avec chaque ressource
    # --------------------------------------------------------

    for _, previous in user_resource_rows.iterrows():

        score = 0.0

        # Même site
        if (
            resource_row["site"]
            == previous["site"]
        ):
            score += 0.20

        # Même sous-site
        if (
            resource_row["sous-site"]
            == previous["sous-site"]
        ):
            score += 0.30

        # Même bibliothèque
        if (
            resource_row["bibliothèque"]
            == previous["bibliothèque"]
        ):
            score += 0.30

        # Même liste
        if (
            resource_row["liste"]
            == previous["liste"]
        ):
            score += 0.20

        best_score = max(
            best_score,
            score
        )

    return min(
        best_score,
        1.0
    )


# ============================================================
# SCORE COLLABORATIF
# ============================================================

def calculate_collaborative_score(
    resource,
    similar_users,
    interactions
):
    """
    Calcule le score collaboratif.

    Plus une ressource est utilisée par des utilisateurs
    similaires, plus son score augmente.

    La similarité des utilisateurs est prise en compte.
    """

    if similar_users.empty:
        return 0.0, 0

    # --------------------------------------------------------
    # Utilisateurs ayant utilisé la ressource
    # --------------------------------------------------------

    users_of_resource = set(
        interactions.loc[
            interactions["resource"] == resource,
            "user"
        ]
    )

    if not users_of_resource:
        return 0.0, 0

    weighted_score = 0.0
    total_similarity = 0.0
    support_users = 0

    # --------------------------------------------------------
    # Parcours des utilisateurs similaires
    # --------------------------------------------------------

    for _, neighbor in similar_users.iterrows():

        similar_user = neighbor[
            "similar_user"
        ]

        similarity = float(
            neighbor["similarity"]
        )

        if similar_user in users_of_resource:

            weighted_score += similarity
            total_similarity += similarity

            support_users += 1

    if total_similarity == 0:
        return 0.0, support_users

    # --------------------------------------------------------
    # Normalisation
    # --------------------------------------------------------

    score = weighted_score

    # La similarité cumulée peut dépasser 1.
    # On normalise pour conserver un score entre 0 et 1.

    score = min(
        score,
        1.0
    )

    return score, support_users


# ============================================================
# SCORE POPULARITE
# ============================================================

def get_popularity_score(
    resource,
    popularity
):
    """
    Retourne le score de popularité d'une ressource.
    """

    row = popularity[
        popularity["resource"] == resource
    ]

    if row.empty:
        return 0.0

    score = float(
        row.iloc[0]["popularity_score"]
    )

    return min(
        max(score, 0.0),
        1.0
    )


# ============================================================
# GENERATION DE L'EXPLICATION
# ============================================================

def generate_explanation(
    support_users,
    collaborative_score,
    context_score,
    popularity_score
):
    """
    Génère une explication lisible de la recommandation.
    """

    reasons = []

    # --------------------------------------------------------
    # Collaboration
    # --------------------------------------------------------

    if support_users >= 3:

        reasons.append(
            f"utilisée par {support_users} "
            "utilisateurs similaires"
        )

    elif support_users == 2:

        reasons.append(
            "utilisée par 2 utilisateurs similaires"
        )

    elif support_users == 1:

        reasons.append(
            "utilisée par 1 utilisateur similaire"
        )

    # --------------------------------------------------------
    # Contexte
    # --------------------------------------------------------

    if context_score >= 0.70:

        reasons.append(
            "forte proximité avec votre environnement"
        )

    elif context_score >= 0.40:

        reasons.append(
            "proximité avec vos ressources actuelles"
        )

    # --------------------------------------------------------
    # Popularité
    # --------------------------------------------------------

    if popularity_score >= 0.80:

        reasons.append(
            "ressource très populaire"
        )

    elif popularity_score >= 0.50:

        reasons.append(
            "ressource régulièrement utilisée"
        )

    # --------------------------------------------------------
    # Cas particulier
    # --------------------------------------------------------

    if not reasons:

        reasons.append(
            "ressource potentiellement pertinente"
        )

    return " ; ".join(
        reasons
    )


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
    """
    Génère les recommandations pour un utilisateur.

    Retourne un DataFrame contenant :

        resource
        site
        sous-site
        bibliothèque
        liste
        collaborative_score
        context_score
        popularity_score
        score
        score_percent
        support_users
        recommendation_type
        explanation
    """

    # ========================================================
    # RESSOURCES DEJA UTILISEES
    # ========================================================

    user_interactions = interactions[
        interactions["user"] == user
    ]

    user_resources = set(
        user_interactions["resource"]
    )

    # ========================================================
    # UTILISATEURS SIMILAIRES
    # ========================================================

    user_neighbors = neighbors[
        neighbors["user"] == user
    ].copy()

    if not user_neighbors.empty:

        user_neighbors = (
            user_neighbors
            .sort_values(
                "similarity",
                ascending=False
            )
            .head(MAX_NEIGHBORS)
        )

    # ========================================================
    # CALCUL DES RECOMMANDATIONS
    # ========================================================

    results = []

    for _, resource_row in resources.iterrows():

        resource = resource_row[
            "resource"
        ]

        # ----------------------------------------------------
        # Ne pas recommander une ressource déjà utilisée
        # ----------------------------------------------------

        if resource in user_resources:
            continue

        # ----------------------------------------------------
        # Score collaboratif
        # ----------------------------------------------------

        collaborative_score, support_users = (
            calculate_collaborative_score(
                resource,
                user_neighbors,
                interactions
            )
        )

        # ----------------------------------------------------
        # Score contextuel
        # ----------------------------------------------------

        context_score = (
            calculate_context_score(
                resource_row,
                user_resources,
                resources
            )
        )

        # ----------------------------------------------------
        # Score popularité
        # ----------------------------------------------------

        popularity_score = (
            get_popularity_score(
                resource,
                popularity
            )
        )

        # ----------------------------------------------------
        # Score final
        # ----------------------------------------------------

        final_score = (

            COLLAB_WEIGHT
            * collaborative_score

            +

            CONTEXT_WEIGHT
            * context_score

            +

            POPULARITY_WEIGHT
            * popularity_score
        )

        # ----------------------------------------------------
        # Type de recommandation
        # ----------------------------------------------------

        if support_users >= 2:

            recommendation_type = (
                "Collaborative + contexte"
            )

        elif support_users == 1:

            recommendation_type = (
                "Collaborative + contexte"
            )

        elif context_score >= 0.40:

            recommendation_type = (
                "Contexte"
            )

        elif popularity_score >= 0.70:

            recommendation_type = (
                "Popularité"
            )

        else:

            recommendation_type = (
                "Hybride"
            )

        # ----------------------------------------------------
        # Explication
        # ----------------------------------------------------

        explanation = generate_explanation(
            support_users,
            collaborative_score,
            context_score,
            popularity_score
        )

        # ----------------------------------------------------
        # Résultat
        # ----------------------------------------------------

        results.append({

            "resource": resource,

            "site": resource_row[
                "site"
            ],

            "sous-site": resource_row[
                "sous-site"
            ],

            "bibliothèque": resource_row[
                "bibliothèque"
            ],

            "liste": resource_row[
                "liste"
            ],

            "collaborative_score":
                collaborative_score,

            "context_score":
                context_score,

            "popularity_score":
                popularity_score,

            "score":
                final_score,

            "score_percent":
                round(
                    final_score * 100,
                    2
                ),

            "support_users":
                support_users,

            "recommendation_type":
                recommendation_type,

            "explanation":
                explanation
        })

    # ========================================================
    # DATAFRAME FINAL
    # ========================================================

    if not results:

        return pd.DataFrame(
            columns=[
                "resource",
                "site",
                "sous-site",
                "bibliothèque",
                "liste",
                "collaborative_score",
                "context_score",
                "popularity_score",
                "score",
                "score_percent",
                "support_users",
                "recommendation_type",
                "explanation"
            ]
        )

    result = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Tri
    # --------------------------------------------------------

    result = (
        result
        .sort_values(
            [
                "score",
                "support_users"
            ],
            ascending=[
                False,
                False
            ]
        )
        .head(top_n)
        .reset_index(drop=True)
    )

    return result


# ============================================================
# AFFICHAGE CONSOLE
# ============================================================

def display_recommendations(
    recommendations
):
    """
    Affiche les recommandations dans la console.
    """

    if recommendations.empty:

        print(
            "\nAucune recommandation disponible."
        )

        return

    columns = [
        "resource",
        "score_percent",
        "support_users",
        "recommendation_type",
        "explanation"
    ]

    print()
    print(
        recommendations[
            columns
        ].to_string(
            index=False
        )
    )


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print(
        "SHAREPOINT RECOMMENDER V3.1"
    )
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
        f"{len(resources):,}"
    )

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
            f"\nUtilisateur '{user}' "
            "introuvable."
        )

        print(
            "\nQuelques utilisateurs disponibles :"
        )

        print(
            sorted(
                available_users
            )[:20]
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
    print(
        f"RECOMMANDATIONS POUR : {user}"
    )
    print("=" * 70)

    display_recommendations(
        recommendations
    )

    print()
    print("=" * 70)