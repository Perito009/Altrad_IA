"""
===============================================================================
                    SHAREPOINT RECOMMENDER - SIMILARITY
===============================================================================

Description
-----------
Ce module calcule les similarités entre utilisateurs à partir de la matrice :

        Utilisateur × Ressource

Une ressource correspond à :

        Site | Sous-site | Bibliothèque | Liste

Méthodes disponibles :

    1. Cosine Similarity
    2. Jaccard Similarity
    3. K-Nearest Neighbors (KNN)

Ces méthodes seront utilisées ensuite par :

        recommender.py

===============================================================================
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


# =============================================================================
# CONFIGURATION
# =============================================================================

PROCESSED_PATH = Path("processed")

USER_RESOURCE_FILE = (
    PROCESSED_PATH / "user_resource.parquet"
)


# =============================================================================
# CHARGEMENT DE LA MATRICE
# =============================================================================

def load_user_resource_matrix(
    path: Path = USER_RESOURCE_FILE,
) -> pd.DataFrame:
    """
    Charge la matrice utilisateur × ressource.

    Parameters
    ----------
    path : Path
        Chemin du fichier Parquet.

    Returns
    -------
    pd.DataFrame
        Matrice d'interactions.

    Raises
    ------
    FileNotFoundError
        Si la matrice n'existe pas.
    """

    if not path.exists():

        raise FileNotFoundError(
            f"Matrice introuvable : {path}. "
            "Lance d'abord preprocessing.py."
        )

    matrix = pd.read_parquet(path)

    return matrix


# =============================================================================
# COSINE SIMILARITY
# =============================================================================

def calculate_cosine_similarity(
    matrix: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcule la similarité cosinus entre tous les utilisateurs.

    La similarité cosinus mesure l'angle entre deux vecteurs.

    Exemple :

        User A -> [1, 1, 0, 0]
        User B -> [1, 1, 1, 0]

    Plus les profils sont proches, plus le score est proche de 1.

    Score :

        1.0 = utilisateurs identiques
        0.0 = aucun élément commun

    Parameters
    ----------
    matrix : pd.DataFrame
        Matrice utilisateur × ressource.

    Returns
    -------
    pd.DataFrame
        Matrice utilisateur × utilisateur.
    """

    similarity = cosine_similarity(matrix)

    similarity_df = pd.DataFrame(
        similarity,
        index=matrix.index,
        columns=matrix.index,
    )

    return similarity_df


# =============================================================================
# JACCARD SIMILARITY
# =============================================================================

def calculate_jaccard_similarity(
    matrix: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcule la similarité de Jaccard entre utilisateurs.

    Jaccard est particulièrement adapté aux données binaires.

    Formule :

        J(A,B) = |A ∩ B| / |A ∪ B|

    Exemple :

        User A possède :

            {A, B, C}

        User B possède :

            {A, B, D}

        Intersection :

            {A, B} = 2

        Union :

            {A, B, C, D} = 4

        Score :

            2 / 4 = 0.5

    Parameters
    ----------
    matrix : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """

    values = matrix.to_numpy(dtype=np.int8)

    number_of_users = values.shape[0]

    similarity = np.zeros(
        (number_of_users, number_of_users),
        dtype=np.float32,
    )

    # -------------------------------------------------------------------------
    # Comparaison de chaque paire d'utilisateurs
    # -------------------------------------------------------------------------

    for i in range(number_of_users):

        user_a = values[i]

        for j in range(i, number_of_users):

            user_b = values[j]

            intersection = np.logical_and(
                user_a,
                user_b,
            ).sum()

            union = np.logical_or(
                user_a,
                user_b,
            ).sum()

            if union == 0:

                score = 0.0

            else:

                score = intersection / union

            similarity[i, j] = score

            similarity[j, i] = score

    return pd.DataFrame(
        similarity,
        index=matrix.index,
        columns=matrix.index,
    )


# =============================================================================
# KNN
# =============================================================================

def calculate_knn(
    matrix: pd.DataFrame,
    n_neighbors: int = 10,
) -> pd.DataFrame:
    """
    Calcule les voisins les plus proches avec KNN.

    La distance utilisée est la distance cosinus.

    Parameters
    ----------
    matrix : pd.DataFrame
        Matrice utilisateur × ressource.

    n_neighbors : int
        Nombre de voisins recherchés.

    Returns
    -------
    pd.DataFrame
        Tableau contenant les utilisateurs similaires.
    """

    # Le nombre maximum de voisins ne peut pas dépasser
    # le nombre d'utilisateurs - 1.
    max_neighbors = max(
        1,
        min(
            n_neighbors,
            len(matrix) - 1,
        ),
    )

    model = NearestNeighbors(
        n_neighbors=max_neighbors + 1,
        metric="cosine",
        algorithm="brute",
    )

    model.fit(matrix.values)

    distances, indices = model.kneighbors(
        matrix.values
    )

    results = []

    users = matrix.index.tolist()

    for row_number, user in enumerate(users):

        for position in range(1, len(indices[row_number])):

            neighbor_index = indices[
                row_number
            ][position]

            neighbor = users[
                neighbor_index
            ]

            distance = distances[
                row_number
            ][position]

            similarity = 1 - distance

            results.append(
                {
                    "user": user,
                    "similar_user": neighbor,
                    "similarity": round(
                        float(similarity),
                        4,
                    ),
                }
            )

    return pd.DataFrame(results)


# =============================================================================
# UTILISATEURS SIMILAIRES - COSINE
# =============================================================================

def get_similar_users_cosine(
    user: str,
    similarity_matrix: pd.DataFrame,
    n: int = 10,
) -> pd.DataFrame:
    """
    Retourne les utilisateurs les plus similaires
    à un utilisateur donné avec Cosine Similarity.

    Parameters
    ----------
    user : str
        Utilisateur recherché.

    similarity_matrix : pd.DataFrame
        Matrice de similarité.

    n : int
        Nombre de résultats.

    Returns
    -------
    pd.DataFrame
    """

    if user not in similarity_matrix.index:

        raise ValueError(
            f"Utilisateur inconnu : {user}"
        )

    scores = (
        similarity_matrix[user]
        .drop(index=user)
        .sort_values(
            ascending=False
        )
        .head(n)
    )

    result = scores.reset_index()

    result.columns = [
        "similar_user",
        "similarity",
    ]

    return result


# =============================================================================
# UTILISATEURS SIMILAIRES - JACCARD
# =============================================================================

def get_similar_users_jaccard(
    user: str,
    similarity_matrix: pd.DataFrame,
    n: int = 10,
) -> pd.DataFrame:
    """
    Retourne les utilisateurs les plus similaires
    avec la méthode Jaccard.
    """

    if user not in similarity_matrix.index:

        raise ValueError(
            f"Utilisateur inconnu : {user}"
        )

    scores = (
        similarity_matrix[user]
        .drop(index=user)
        .sort_values(
            ascending=False
        )
        .head(n)
    )

    result = scores.reset_index()

    result.columns = [
        "similar_user",
        "similarity",
    ]

    return result


# =============================================================================
# COMPARAISON DES MÉTHODES
# =============================================================================

def compare_user_similarity(
    user: str,
    cosine_matrix: pd.DataFrame,
    jaccard_matrix: pd.DataFrame,
    n: int = 10,
) -> pd.DataFrame:
    """
    Compare Cosine et Jaccard pour un utilisateur.

    Returns
    -------
    pd.DataFrame

    Exemple :

        similar_user | cosine | jaccard
        --------------------------------
        user_1       | 0.91   | 0.82
        user_2       | 0.87   | 0.79
    """

    if user not in cosine_matrix.index:

        raise ValueError(
            f"Utilisateur inconnu : {user}"
        )

    cosine_scores = (
        cosine_matrix[user]
        .drop(index=user)
    )

    jaccard_scores = (
        jaccard_matrix[user]
        .drop(index=user)
    )

    result = pd.DataFrame(
        {
            "similar_user": cosine_scores.index,
            "cosine": cosine_scores.values,
            "jaccard": jaccard_scores.values,
        }
    )

    # Score hybride simple.
    result["hybrid_score"] = (
        0.6 * result["cosine"]
        + 0.4 * result["jaccard"]
    )

    result = result.sort_values(
        "hybrid_score",
        ascending=False,
    )

    return result.head(n)


# =============================================================================
# SAUVEGARDE DES MATRICES DE SIMILARITÉ
# =============================================================================

def save_similarity_matrix(
    matrix: pd.DataFrame,
    filename: str,
) -> None:
    """
    Sauvegarde une matrice de similarité au format Parquet.
    """

    output_path = (
        PROCESSED_PATH / filename
    )

    matrix.to_parquet(
        output_path
    )


# =============================================================================
# PIPELINE DE SIMILARITÉ
# =============================================================================

def run_similarity_pipeline() -> dict:
    """
    Exécute le calcul complet des similarités.

    Returns
    -------
    dict
        Contient :

            cosine
            jaccard
            knn
    """

    print(
        "\n=========================================="
    )

    print(
        "CALCUL DES SIMILARITÉS"
    )

    print(
        "=========================================="
    )

    # -------------------------------------------------------------------------
    # Chargement
    # -------------------------------------------------------------------------

    matrix = load_user_resource_matrix()

    print(
        f"Utilisateurs : {len(matrix)}"
    )

    print(
        f"Ressources : {len(matrix.columns)}"
    )

    # -------------------------------------------------------------------------
    # Cosine
    # -------------------------------------------------------------------------

    print(
        "\nCalcul Cosine Similarity..."
    )

    cosine_matrix = (
        calculate_cosine_similarity(
            matrix
        )
    )

    save_similarity_matrix(
        cosine_matrix,
        "cosine_similarity.parquet",
    )

    # -------------------------------------------------------------------------
    # Jaccard
    # -------------------------------------------------------------------------

    print(
        "Calcul Jaccard Similarity..."
    )

    jaccard_matrix = (
        calculate_jaccard_similarity(
            matrix
        )
    )

    save_similarity_matrix(
        jaccard_matrix,
        "jaccard_similarity.parquet",
    )

    # -------------------------------------------------------------------------
    # KNN
    # -------------------------------------------------------------------------

    print(
        "Calcul KNN..."
    )

    knn_results = calculate_knn(
        matrix,
        n_neighbors=10,
    )

    knn_results.to_parquet(
        PROCESSED_PATH
        / "knn_neighbors.parquet",
        index=False,
    )

    print(
        "\nCalcul terminé."
    )

    return {
        "cosine": cosine_matrix,
        "jaccard": jaccard_matrix,
        "knn": knn_results,
    }


# =============================================================================
# TEST DU MODULE
# =============================================================================

if __name__ == "__main__":

    results = run_similarity_pipeline()

    print(
        "\nFichiers générés :"
    )

    print(
        "  - cosine_similarity.parquet"
    )

    print(
        "  - jaccard_similarity.parquet"
    )

    print(
        "  - knn_neighbors.parquet"
    )