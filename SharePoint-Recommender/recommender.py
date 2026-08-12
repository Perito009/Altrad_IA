"""
===============================================================================
                    SHAREPOINT RECOMMENDER
                         RECOMMENDER ENGINE
===============================================================================

Objectif
--------
Générer des recommandations personnalisées de ressources SharePoint.

Principe
--------
Le système utilise une approche collaborative basée sur les utilisateurs.

Exemple :

Utilisateur A
    ├── Ressource 1
    ├── Ressource 2
    └── Ressource 3

Utilisateur B
    ├── Ressource 1
    ├── Ressource 2
    └── Ressource 4

Utilisateur A et B sont similaires.

On peut donc recommander :

    Ressource 4

à l'utilisateur A.

===============================================================================
"""


# =============================================================================
# IMPORTS
# =============================================================================

from pathlib import Path

import pandas as pd
import numpy as np


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


PROCESSED_DIR = (
    BASE_DIR
    / "processed"
)


USER_RESOURCE_PATH = (
    PROCESSED_DIR
    / "user_resource.parquet"
)


CLEAN_DATA_PATH = (
    PROCESSED_DIR
    / "clean_data.parquet"
)


COSINE_PATH = (
    PROCESSED_DIR
    / "cosine_similarity.parquet"
)


JACCARD_PATH = (
    PROCESSED_DIR
    / "jaccard_similarity.parquet"
)


KNN_PATH = (
    PROCESSED_DIR
    / "knn_neighbors.parquet"
)


# =============================================================================
# CHARGEMENT DES DONNÉES
# =============================================================================

def load_user_resource_matrix():
    """
    Charge la matrice utilisateur × ressource.

    Returns
    -------
    pandas.DataFrame

        Index :
            utilisateurs

        Colonnes :
            ressources

        Valeurs :
            0 ou 1
    """

    if not USER_RESOURCE_PATH.exists():

        raise FileNotFoundError(
            f"Matrice utilisateur × ressource introuvable : "
            f"{USER_RESOURCE_PATH}"
        )


    matrix = pd.read_parquet(
        USER_RESOURCE_PATH
    )


    return matrix


# =============================================================================
# CHARGEMENT DE LA SIMILARITÉ COSINE
# =============================================================================

def load_cosine_similarity():
    """
    Charge la matrice de similarité Cosine.

    Returns
    -------
    pandas.DataFrame
    """

    if not COSINE_PATH.exists():

        raise FileNotFoundError(
            f"Fichier de similarité Cosine introuvable : "
            f"{COSINE_PATH}"
        )


    similarity = pd.read_parquet(
        COSINE_PATH
    )


    return similarity


# =============================================================================
# CHARGEMENT DE LA SIMILARITÉ JACCARD
# =============================================================================

def load_jaccard_similarity():
    """
    Charge la matrice de similarité Jaccard.

    Returns
    -------
    pandas.DataFrame
    """

    if not JACCARD_PATH.exists():

        raise FileNotFoundError(
            f"Fichier Jaccard introuvable : "
            f"{JACCARD_PATH}"
        )


    return pd.read_parquet(
        JACCARD_PATH
    )


# =============================================================================
# CHARGEMENT DES VOISINS KNN
# =============================================================================

def load_knn_neighbors():
    """
    Charge les voisins calculés avec KNN.

    Returns
    -------
    pandas.DataFrame
    """

    if not KNN_PATH.exists():

        raise FileNotFoundError(
            f"Fichier KNN introuvable : "
            f"{KNN_PATH}"
        )


    return pd.read_parquet(
        KNN_PATH
    )


# =============================================================================
# CHARGEMENT DES DONNÉES PROPRES
# =============================================================================

def load_clean_data():
    """
    Charge le dataset nettoyé.

    Ce dataset permet notamment de récupérer
    les informations détaillées des ressources.
    """

    if not CLEAN_DATA_PATH.exists():

        raise FileNotFoundError(
            f"Dataset nettoyé introuvable : "
            f"{CLEAN_DATA_PATH}"
        )


    return pd.read_parquet(
        CLEAN_DATA_PATH
    )


# =============================================================================
# RESSOURCES D'UN UTILISATEUR
# =============================================================================

def get_user_resources(
    user,
    user_resource_matrix=None
):
    """
    Retourne les ressources déjà utilisées par un utilisateur.

    Parameters
    ----------
    user : str
        Utilisateur.

    user_resource_matrix : DataFrame, optional
        Matrice utilisateur × ressource.

    Returns
    -------
    list
        Liste des ressources.
    """

    if user_resource_matrix is None:

        user_resource_matrix = (
            load_user_resource_matrix()
        )


    if user not in user_resource_matrix.index:

        return []


    user_row = (
        user_resource_matrix
        .loc[user]
    )


    resources = (
        user_row[
            user_row > 0
        ]
        .index
        .tolist()
    )


    return resources


# =============================================================================
# UTILISATEURS SIMILAIRES
# =============================================================================

def get_similar_users(
    user,
    similarity_matrix=None,
    n=10
):
    """
    Retourne les utilisateurs les plus similaires.

    Parameters
    ----------
    user : str
        Utilisateur cible.

    similarity_matrix : DataFrame
        Matrice de similarité.

    n : int
        Nombre de voisins.

    Returns
    -------
    pandas.DataFrame

        Colonnes :

        user
        similarity
    """

    if similarity_matrix is None:

        similarity_matrix = (
            load_cosine_similarity()
        )


    if user not in similarity_matrix.index:

        return pd.DataFrame(
            columns=[
                "user",
                "similarity"
            ]
        )


    similarities = (
        similarity_matrix
        .loc[user]
        .copy()
    )


    # -------------------------------------------------------------------------
    # Suppression de l'utilisateur lui-même
    # -------------------------------------------------------------------------

    if user in similarities.index:

        similarities = (
            similarities
            .drop(user)
        )


    # -------------------------------------------------------------------------
    # Tri décroissant
    # -------------------------------------------------------------------------

    similarities = (
        similarities
        .sort_values(
            ascending=False
        )
    )


    # -------------------------------------------------------------------------
    # Suppression des similarités nulles
    # -------------------------------------------------------------------------

    similarities = (
        similarities[
            similarities > 0
        ]
    )


    similarities = (
        similarities
        .head(n)
    )


    result = pd.DataFrame(
        {
            "user": similarities.index,
            "similarity": similarities.values,
        }
    )


    return result.reset_index(
        drop=True
    )


# =============================================================================
# RECOMMANDATION
# =============================================================================

def recommend_resources(
    user,
    similarity_matrix=None,
    user_resource_matrix=None,
    n_users=10,
    n_recommendations=10
):
    """
    Génère des recommandations personnalisées.

    Méthode
    -------
    1. Trouver les utilisateurs similaires.
    2. Récupérer leurs ressources.
    3. Exclure les ressources déjà utilisées.
    4. Pondérer chaque ressource par la similarité
       de l'utilisateur qui la possède.
    5. Classer les ressources.

    Parameters
    ----------
    user : str
        Utilisateur cible.

    similarity_matrix : DataFrame
        Matrice Cosine.

    user_resource_matrix : DataFrame
        Matrice utilisateur × ressource.

    n_users : int
        Nombre d'utilisateurs similaires.

    n_recommendations : int
        Nombre de recommandations.

    Returns
    -------
    pandas.DataFrame
    """

    if similarity_matrix is None:

        similarity_matrix = (
            load_cosine_similarity()
        )


    if user_resource_matrix is None:

        user_resource_matrix = (
            load_user_resource_matrix()
        )


    # =========================================================================
    # 1. RESSOURCES DÉJÀ UTILISÉES
    # =========================================================================

    current_resources = set(
        get_user_resources(
            user,
            user_resource_matrix
        )
    )


    # =========================================================================
    # 2. UTILISATEURS SIMILAIRES
    # =========================================================================

    similar_users = get_similar_users(
        user,
        similarity_matrix,
        n=n_users
    )


    if similar_users.empty:

        return pd.DataFrame(
            columns=[
                "resource",
                "score",
                "support_count",
                "average_similarity",
                "score_percent",
            ]
        )


    # =========================================================================
    # 3. CALCUL DES SCORES
    # =========================================================================

    resource_scores = {}

    resource_support = {}

    resource_similarities = {}


    for _, row in similar_users.iterrows():

        similar_user = row[
            "user"
        ]

        similarity = float(
            row[
                "similarity"
            ]
        )


        # -------------------------------------------------------------
        # Ressources du voisin
        # -------------------------------------------------------------

        neighbor_resources = (
            get_user_resources(
                similar_user,
                user_resource_matrix
            )
        )


        for resource in neighbor_resources:

            # ---------------------------------------------------------
            # On ne recommande pas une ressource déjà utilisée.
            # ---------------------------------------------------------

            if resource in current_resources:

                continue


            # ---------------------------------------------------------
            # Score pondéré par la similarité.
            # ---------------------------------------------------------

            resource_scores[
                resource
            ] = (
                resource_scores.get(
                    resource,
                    0
                )
                + similarity
            )


            # ---------------------------------------------------------
            # Nombre d'utilisateurs similaires
            # utilisant cette ressource.
            # ---------------------------------------------------------

            resource_support[
                resource
            ] = (
                resource_support.get(
                    resource,
                    0
                )
                + 1
            )


            # ---------------------------------------------------------
            # Conservation des similarités.
            # ---------------------------------------------------------

            if resource not in resource_similarities:

                resource_similarities[
                    resource
                ] = []


            resource_similarities[
                resource
            ].append(
                similarity
            )


    # =========================================================================
    # 4. AUCUNE RECOMMANDATION
    # =========================================================================

    if not resource_scores:

        return pd.DataFrame(
            columns=[
                "resource",
                "score",
                "support_count",
                "average_similarity",
                "score_percent",
            ]
        )


    # =========================================================================
    # 5. CONSTRUCTION DU DATAFRAME
    # =========================================================================

    recommendations = []


    for resource, score in resource_scores.items():

        similarities = (
            resource_similarities[
                resource
            ]
        )


        average_similarity = (
            np.mean(
                similarities
            )
        )


        recommendations.append(
            {
                "resource": resource,

                "score": score,

                "support_count": (
                    resource_support[
                        resource
                    ]
                ),

                "average_similarity": (
                    average_similarity
                ),
            }
        )


    recommendations = pd.DataFrame(
        recommendations
    )


    # =========================================================================
    # 6. NORMALISATION DU SCORE
    # =========================================================================

    max_score = (
        recommendations[
            "score"
        ].max()
    )


    if max_score > 0:

        recommendations[
            "score_percent"
        ] = (
            recommendations[
                "score"
            ]
            / max_score
            * 100
        )

    else:

        recommendations[
            "score_percent"
        ] = 0


    # =========================================================================
    # 7. TRI FINAL
    # =========================================================================

    recommendations = (
        recommendations
        .sort_values(
            by=[
                "score",
                "support_count",
                "average_similarity",
            ],
            ascending=False
        )
    )


    # =========================================================================
    # 8. TOP N
    # =========================================================================

    recommendations = (
        recommendations
        .head(
            n_recommendations
        )
        .reset_index(
            drop=True
        )
    )


    return recommendations


# =============================================================================
# POPULARITÉ
# =============================================================================

def popular_resources(
    user,
    user_resource_matrix=None,
    n=10
):
    """
    Retourne les ressources les plus populaires.

    Cette méthode constitue une baseline simple.

    Une ressource populaire est une ressource utilisée
    par beaucoup d'utilisateurs.

    Parameters
    ----------
    user : str

    user_resource_matrix : DataFrame

    n : int

    Returns
    -------
    pandas.DataFrame
    """

    if user_resource_matrix is None:

        user_resource_matrix = (
            load_user_resource_matrix()
        )


    current_resources = set(
        get_user_resources(
            user,
            user_resource_matrix
        )
    )


    # -------------------------------------------------------------------------
    # Nombre d'utilisateurs par ressource
    # -------------------------------------------------------------------------

    popularity = (
        user_resource_matrix
        .sum(axis=0)
    )


    popularity = (
        popularity
        .sort_values(
            ascending=False
        )
    )


    # -------------------------------------------------------------------------
    # Suppression des ressources déjà utilisées
    # -------------------------------------------------------------------------

    popularity = popularity[
        ~popularity.index.isin(
            current_resources
        )
    ]


    popularity = (
        popularity
        .head(n)
    )


    result = pd.DataFrame(
        {
            "resource": popularity.index,
            "users_count": popularity.values,
        }
    )


    return result.reset_index(
        drop=True
    )


# =============================================================================
# EXPLICATION D'UNE RECOMMANDATION
# =============================================================================

def explain_recommendation(
    resource,
    user,
    similarity_matrix=None,
    user_resource_matrix=None,
    n_users=10
):
    """
    Explique pourquoi une ressource est recommandée.

    Exemple :

        La ressource X est recommandée parce que :

        - Fabien est similaire à Yann à 82 %
        - Isabelle est similaire à Yann à 76 %
        - ces deux utilisateurs utilisent X

    Returns
    -------
    dict
    """

    if similarity_matrix is None:

        similarity_matrix = (
            load_cosine_similarity()
        )


    if user_resource_matrix is None:

        user_resource_matrix = (
            load_user_resource_matrix()
        )


    similar_users = get_similar_users(
        user,
        similarity_matrix,
        n=n_users
    )


    supporting_users = []


    for _, row in similar_users.iterrows():

        similar_user = row[
            "user"
        ]

        similarity = float(
            row[
                "similarity"
            ]
        )


        resources = get_user_resources(
            similar_user,
            user_resource_matrix
        )


        if resource in resources:

            supporting_users.append(
                {
                    "user": similar_user,
                    "similarity": similarity,
                }
            )


    supporting_users = sorted(
        supporting_users,
        key=lambda x: x[
            "similarity"
        ],
        reverse=True
    )


    return {
        "resource": resource,

        "support_count": len(
            supporting_users
        ),

        "supporting_users": (
            supporting_users
        ),
    }


# =============================================================================
# TEST DU MODULE
# =============================================================================

if __name__ == "__main__":

    print()
    print(
        "=" * 70
    )

    print(
        "TEST DU SYSTÈME DE RECOMMANDATION"
    )

    print(
        "=" * 70
    )


    # -------------------------------------------------------------------------
    # Chargement
    # -------------------------------------------------------------------------

    matrix = (
        load_user_resource_matrix()
    )

    cosine = (
        load_cosine_similarity()
    )


    print(
        f"\n👥 Utilisateurs : {len(matrix.index)}"
    )

    print(
        f"📚 Ressources : {len(matrix.columns)}"
    )


    # -------------------------------------------------------------------------
    # Sélection automatique d'un utilisateur
    #
    # On prend ici un utilisateur ayant
    # au moins une interaction.
    # -------------------------------------------------------------------------

    users_with_resources = (
        matrix.sum(axis=1)
        .sort_values(
            ascending=False
        )
    )


    selected_user = (
        users_with_resources
        .index[0]
    )


    print(
        f"\n👤 Utilisateur de test : "
        f"{selected_user}"
    )


    # -------------------------------------------------------------------------
    # Ressources actuelles
    # -------------------------------------------------------------------------

    current = get_user_resources(
        selected_user,
        matrix
    )


    print(
        f"\n📚 Ressources actuelles : "
        f"{len(current)}"
    )


    # -------------------------------------------------------------------------
    # Utilisateurs similaires
    # -------------------------------------------------------------------------

    similar = get_similar_users(
        selected_user,
        cosine,
        n=10
    )


    print(
        "\n👥 Utilisateurs similaires :"
    )


    print(
        similar.to_string(
            index=False
        )
    )


    # -------------------------------------------------------------------------
    # Recommandations
    # -------------------------------------------------------------------------

    recommendations = (
        recommend_resources(
            user=selected_user,

            similarity_matrix=cosine,

            user_resource_matrix=matrix,

            n_users=10,

            n_recommendations=10,
        )
    )


    print(
        "\n🎯 RECOMMANDATIONS :"
    )


    if recommendations.empty:

        print(
            "Aucune recommandation disponible."
        )

    else:

        print(
            recommendations.to_string(
                index=False
            )
        )


    print()
    print(
        "=" * 70
    )

    print(
        "TEST TERMINÉ"
    )

    print(
        "=" * 70
    )