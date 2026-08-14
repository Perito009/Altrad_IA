"""
============================================================
SHAREPOINT RECOMMENDER - RECOMMENDER V3
============================================================

Système de recommandation hybride pour SharePoint.

Le système combine 3 sources d'information :

1. Filtrage collaboratif
   -> utilisateurs similaires

2. Popularité
   -> ressources utilisées par beaucoup d'utilisateurs

3. Contexte SharePoint
   -> site, sous-site, bibliothèque et liste

Le résultat est un score hybride.

Formule :

    score_final =
        60% collaboratif
        25% popularité
        15% contexte

Fichiers nécessaires :

    processed/interactions.parquet
    processed/resources.parquet
    processed/user_neighbors.parquet
    processed/resource_popularity.parquet
"""

from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION DES CHEMINS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
PROCESSED_DIR = PROJECT_DIR / "processed"

INTERACTIONS_FILE = PROCESSED_DIR / "interactions.parquet"
RESOURCES_FILE = PROCESSED_DIR / "resources.parquet"
NEIGHBORS_FILE = PROCESSED_DIR / "user_neighbors.parquet"
POPULARITY_FILE = PROCESSED_DIR / "resource_popularity.parquet"


# ============================================================
# POIDS DU SYSTÈME
# ============================================================

# Le filtrage collaboratif est volontairement majoritaire.
COLLABORATIVE_WEIGHT = 0.60

# La popularité sert de complément.
POPULARITY_WEIGHT = 0.25

# Le contexte permet de favoriser les ressources
# appartenant à des espaces SharePoint proches.
CONTEXT_WEIGHT = 0.15


# ============================================================
# CHARGEMENT DES DONNÉES
# ============================================================

def load_data():
    """
    Charge les différents fichiers Parquet nécessaires
    au système de recommandation.
    """

    # Vérification des fichiers
    files = {
        "interactions": INTERACTIONS_FILE,
        "resources": RESOURCES_FILE,
        "neighbors": NEIGHBORS_FILE,
        "popularity": POPULARITY_FILE,
    }

    for name, path in files.items():

        if not path.exists():
            raise FileNotFoundError(
                f"Fichier '{name}' introuvable : {path}"
            )

    # Chargement
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
        popularity,
    )


# ============================================================
# SCORE COLLABORATIF
# ============================================================

def calculate_collaborative_score(
    user,
    interactions,
    neighbors,
):
    """
    Calcule les ressources recommandées grâce aux
    utilisateurs similaires.

    Exemple :

        Fabien
           ↓
        utilisateurs similaires
           ↓
        ressources utilisées par ces utilisateurs
           ↓
        score collaboratif

    Plus la similarité du voisin est élevée,
    plus son interaction influence la recommandation.
    """

    # --------------------------------------------------------
    # Récupération des voisins
    # --------------------------------------------------------

    user_neighbors = neighbors[
        neighbors["user"] == user
    ].copy()

    if user_neighbors.empty:

        return pd.DataFrame(
            columns=[
                "resource",
                "collaborative_score",
                "support_users",
            ]
        )

    # --------------------------------------------------------
    # Ressources utilisées par les voisins
    # --------------------------------------------------------

    neighbor_interactions = interactions.merge(
        user_neighbors[
            [
                "similar_user",
                "similarity",
            ]
        ],
        left_on="user",
        right_on="similar_user",
        how="inner",
    )

    if neighbor_interactions.empty:

        return pd.DataFrame(
            columns=[
                "resource",
                "collaborative_score",
                "support_users",
            ]
        )

    # --------------------------------------------------------
    # Calcul du score
    # --------------------------------------------------------

    collaborative = (
        neighbor_interactions
        .groupby("resource")
        .agg(
            collaborative_score=(
                "similarity",
                "sum",
            ),
            support_users=(
                "similar_user",
                "nunique",
            ),
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Normalisation entre 0 et 1
    # --------------------------------------------------------

    max_score = collaborative[
        "collaborative_score"
    ].max()

    if max_score > 0:

        collaborative[
            "collaborative_score"
        ] = (
            collaborative[
                "collaborative_score"
            ]
            / max_score
        )

    return collaborative


# ============================================================
# SCORE DE POPULARITÉ
# ============================================================

def calculate_popularity_score(
    popularity,
):
    """
    Prépare le score de popularité.

    popularity.py produit déjà un score compris entre 0 et 1.
    """

    return popularity[
        [
            "resource",
            "popularity_score",
        ]
    ].copy()


# ============================================================
# SCORE CONTEXTUEL
# ============================================================

def calculate_context_score(
    user,
    interactions,
    resources,
):
    """
    Calcule la proximité entre les ressources déjà utilisées
    par l'utilisateur et les ressources candidates.

    Pondération du contexte :

        Site          : 40 %
        Sous-site     : 30 %
        Bibliothèque  : 20 %
        Liste         : 10 %
    """

    # --------------------------------------------------------
    # Ressources déjà utilisées
    # --------------------------------------------------------

    user_interactions = interactions[
        interactions["user"] == user
    ]

    if user_interactions.empty:

        return pd.DataFrame(
            columns=[
                "resource",
                "context_score",
            ]
        )

    used_resources = resources[
        resources["resource"].isin(
            user_interactions["resource"]
        )
    ]

    if used_resources.empty:

        return pd.DataFrame(
            columns=[
                "resource",
                "context_score",
            ]
        )

    # --------------------------------------------------------
    # Contextes déjà utilisés
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
    # Calcul
    # --------------------------------------------------------

    context_scores = []

    for _, resource in resources.iterrows():

        score = 0.0

        # Même site
        if resource["site"] in user_sites:
            score += 0.40

        # Même sous-site
        if resource["sous-site"] in user_subsites:
            score += 0.30

        # Même bibliothèque
        if resource["bibliothèque"] in user_libraries:
            score += 0.20

        # Même liste
        if resource["liste"] in user_lists:
            score += 0.10

        context_scores.append(
            {
                "resource": resource["resource"],
                "context_score": score,
            }
        )

    return pd.DataFrame(
        context_scores
    )


# ============================================================
# TYPE DE RECOMMANDATION
# ============================================================

def determine_recommendation_type(row):
    """
    Détermine la principale raison de la recommandation.
    """

    collaborative = row["collaborative_score"]
    context = row["context_score"]
    popularity = row["popularity_score"]

    # Cas collaboratif
    if (
        collaborative > 0
        and context > 0
    ):
        return "Collaboratif + contexte"

    if collaborative > 0:
        return "Collaboratif"

    # Cas contextuel
    if context > 0:
        return "Contexte"

    # Dernier recours
    if popularity > 0:
        return "Popularité"

    return "Inconnu"


# ============================================================
# MOTEUR DE RECOMMANDATION
# ============================================================

def recommend(
    user,
    interactions,
    resources,
    neighbors,
    popularity,
    top_n=10,
):
    """
    Retourne les meilleures recommandations pour un utilisateur.
    """

    # ========================================================
    # RESSOURCES DÉJÀ UTILISÉES
    # ========================================================

    used_resources = set(
        interactions[
            interactions["user"] == user
        ]["resource"]
    )

    # ========================================================
    # SCORE COLLABORATIF
    # ========================================================

    collaborative = calculate_collaborative_score(
        user=user,
        interactions=interactions,
        neighbors=neighbors,
    )

    # ========================================================
    # SCORE POPULARITÉ
    # ========================================================

    popularity_scores = calculate_popularity_score(
        popularity
    )

    # ========================================================
    # SCORE CONTEXTE
    # ========================================================

    context = calculate_context_score(
        user=user,
        interactions=interactions,
        resources=resources,
    )

    # ========================================================
    # TABLE DE BASE
    # ========================================================

    recommendations = resources.copy()

    # --------------------------------------------------------
    # Ajout du score collaboratif
    # --------------------------------------------------------

    recommendations = recommendations.merge(
        collaborative,
        on="resource",
        how="left",
    )

    # --------------------------------------------------------
    # Ajout de la popularité
    # --------------------------------------------------------

    recommendations = recommendations.merge(
        popularity_scores,
        on="resource",
        how="left",
    )

    # --------------------------------------------------------
    # Ajout du contexte
    # --------------------------------------------------------

    recommendations = recommendations.merge(
        context,
        on="resource",
        how="left",
    )

    # ========================================================
    # VALEURS MANQUANTES
    # ========================================================

    recommendations[
        "collaborative_score"
    ] = recommendations[
        "collaborative_score"
    ].fillna(0.0)

    recommendations[
        "popularity_score"
    ] = recommendations[
        "popularity_score"
    ].fillna(0.0)

    recommendations[
        "context_score"
    ] = recommendations[
        "context_score"
    ].fillna(0.0)

    recommendations[
        "support_users"
    ] = recommendations[
        "support_users"
    ].fillna(0)

    # ========================================================
    # SCORE HYBRIDE
    # ========================================================

    recommendations["score"] = (

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
    # EXCLUSION DES RESSOURCES DÉJÀ UTILISÉES
    # ========================================================

    recommendations = recommendations[
        ~recommendations[
            "resource"
        ].isin(used_resources)
    ].copy()

    # ========================================================
    # TRI
    # ========================================================

    recommendations = (
        recommendations
        .sort_values(
            by="score",
            ascending=False,
        )
        .head(top_n)
        .copy()
    )

    # ========================================================
    # SCORE EN POURCENTAGE
    # ========================================================

    recommendations[
        "score_percent"
    ] = (
        recommendations["score"] * 100
    ).round(2)

    # ========================================================
    # TYPE
    # ========================================================

    recommendations[
        "recommendation_type"
    ] = recommendations.apply(
        determine_recommendation_type,
        axis=1,
    )

    # ========================================================
    # COLONNES FINALES
    # ========================================================

    return recommendations[
        [
            "resource",
            "site",
            "sous-site",
            "bibliothèque",
            "liste",
            "score_percent",
            "support_users",
            "recommendation_type",
        ]
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
    # Chargement des données
    # --------------------------------------------------------

    (
        interactions,
        resources,
        neighbors,
        popularity,
    ) = load_data()

    print(
        f"\nUtilisateurs   : "
        f"{interactions['user'].nunique():,}"
    )

    print(
        f"Ressources     : "
        f"{resources['resource'].nunique():,}"
    )

    print(
        f"Interactions   : "
        f"{len(interactions):,}"
    )

    # --------------------------------------------------------
    # Utilisateur
    # --------------------------------------------------------

    user = input(
        "\nUtilisateur : "
    ).strip().lower()

    # --------------------------------------------------------
    # Vérification utilisateur
    # --------------------------------------------------------

    available_users = set(
        interactions["user"]
    )

    if user not in available_users:

        print(
            f"\n❌ Utilisateur '{user}' introuvable."
        )

        print(
            "\nQuelques utilisateurs disponibles :"
        )

        print(
            ", ".join(
                sorted(
                    available_users
                )[:20]
            )
        )

        raise SystemExit(1)

    # --------------------------------------------------------
    # Recommandations
    # --------------------------------------------------------

    recommendations = recommend(
        user=user,
        interactions=interactions,
        resources=resources,
        neighbors=neighbors,
        popularity=popularity,
        top_n=10,
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