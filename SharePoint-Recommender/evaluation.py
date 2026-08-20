"""
======================================================================
SHAREPOINT RECOMMENDER - EVALUATION V4
======================================================================

Evaluation propre du système de recommandation.

Correction principale par rapport à l'ancienne version :


    MAINTENANT :
        interactions complètes
            ↓
        masquage des ressources de test
            ↓
        reconstruction du profil utilisateur
            ↓
        recalcul dynamique des voisins
            ↓
        génération des recommandations
            ↓
        comparaison avec les ressources masquées


======================================================================
"""

from pathlib import Path
import math
import random

import numpy as np
import pandas as pd


# ======================================================================
# CONFIGURATION
# ======================================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
PROCESSED_DIR = PROJECT_DIR / "processed"

INTERACTIONS_FILE = (
    PROCESSED_DIR / "interactions.parquet"
)


# ======================================================================
# PARAMETRES
# ======================================================================

# Nombre maximum d'utilisateurs évalués
MAX_USERS = 100

# Pourcentage d'interactions cachées pour le test
TEST_RATIO = 0.20

# Nombre de recommandations produites
TOP_K = 10

# Nombre maximum de voisins utilisés
TOP_NEIGHBORS = 30

# Minimum d'interactions pour évaluer un utilisateur
MIN_INTERACTIONS = 5

# Graine permettant d'obtenir les mêmes résultats
RANDOM_SEED = 42


# ======================================================================
# POIDS DU RECOMMANDER V4
# ======================================================================

NEIGHBOR_WEIGHT = 0.70
POPULARITY_WEIGHT = 0.20
SUPPORT_WEIGHT = 0.10


# ======================================================================
# CHARGEMENT
# ======================================================================

def load_interactions():
    """
    Charge interactions.parquet.

    Structure attendue :

        user
        resource
    """

    if not INTERACTIONS_FILE.exists():

        raise FileNotFoundError(
            "\nFichier introuvable : "
            f"{INTERACTIONS_FILE}\n"
            "\nLance d'abord preprocessing.py."
        )

    interactions = pd.read_parquet(
        INTERACTIONS_FILE
    )

    required_columns = {
        "user",
        "resource"
    }

    missing = (
        required_columns
        - set(interactions.columns)
    )

    if missing:

        raise ValueError(
            "Colonnes manquantes : "
            f"{sorted(missing)}"
        )

    interactions = (
        interactions[
            [
                "user",
                "resource"
            ]
        ]
        .dropna()
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return interactions


# ======================================================================
# PROFILS UTILISATEURS
# ======================================================================

def build_profiles(interactions):
    """
    Construit :

        utilisateur -> ensemble de ressources
    """

    return (
        interactions
        .groupby("user")["resource"]
        .apply(set)
        .to_dict()
    )


# ======================================================================
# JACCARD
# ======================================================================

def calculate_jaccard(
    resources_a,
    resources_b
):
    """
    Jaccard :

        |A intersection B|
        ------------------
          |A union B|
    """

    union = resources_a | resources_b

    if not union:
        return 0.0

    intersection = (
        resources_a
        & resources_b
    )

    return (
        len(intersection)
        / len(union)
    )


# ======================================================================
# CONTAINMENT
# ======================================================================

def calculate_containment(
    resources_a,
    resources_b
):
    """
    Containment utilisé dans similarity.py V4 :

        |A intersection B|
        ------------------
          min(|A|, |B|)
    """

    minimum_size = min(
        len(resources_a),
        len(resources_b)
    )

    if minimum_size == 0:
        return 0.0

    intersection = (
        resources_a
        & resources_b
    )

    return (
        len(intersection)
        / minimum_size
    )


# ======================================================================
# SIMILARITE
# ======================================================================

def calculate_similarity(
    resources_a,
    resources_b
):
    """
    Même logique que similarity.py V4.

    Score :

        60 % Jaccard
        40 % Containment
    """

    jaccard = calculate_jaccard(
        resources_a,
        resources_b
    )

    containment = calculate_containment(
        resources_a,
        resources_b
    )

    similarity = (
        0.60 * jaccard
        +
        0.40 * containment
    )

    return (
        similarity,
        jaccard,
        containment
    )


# ======================================================================
# RECHERCHE DYNAMIQUE DES VOISINS
# ======================================================================

def find_neighbors(
    target_user,
    target_resources,
    profiles
):
    """
    Recherche les utilisateurs similaires APRES le masquage.

    Point essentiel :

    new_resources est recalculé dynamiquement avec :

        ressources_voisin
        -
        ressources_train_utilisateur

    On ne réutilise PAS la colonne new_resources
    de user_neighbors.parquet.
    """

    neighbors = []

    for other_user, other_resources in profiles.items():

        # Ne pas comparer l'utilisateur à lui-même
        if other_user == target_user:
            continue

        # Ressources réellement nouvelles pour
        # l'utilisateur dans son profil TRAIN
        new_resources = (
            other_resources
            - target_resources
        )

        # Un voisin incapable d'apporter quoi que
        # ce soit n'est pas utile à la recommandation.
        if not new_resources:
            continue

        common_resources = (
            target_resources
            & other_resources
        )

        # Pas de ressource commune = pas de similarité
        if not common_resources:
            continue

        (
            similarity,
            jaccard,
            containment
        ) = calculate_similarity(
            target_resources,
            other_resources
        )

        if similarity <= 0:
            continue

        neighbors.append(
            {
                "similar_user": other_user,
                "similarity": similarity,
                "jaccard": jaccard,
                "containment": containment,
                "common_resources": len(
                    common_resources
                ),
                "new_resources": len(
                    new_resources
                )
            }
        )

    if not neighbors:

        return pd.DataFrame()

    neighbors = pd.DataFrame(
        neighbors
    )

    # --------------------------------------------------------------
    # Classement
    # --------------------------------------------------------------

    neighbors = neighbors.sort_values(
        by=[
            "similarity",
            "new_resources",
            "common_resources"
        ],
        ascending=[
            False,
            False,
            False
        ]
    )

    return neighbors.head(
        TOP_NEIGHBORS
    )


# ======================================================================
# POPULARITE DYNAMIQUE
# ======================================================================

def calculate_popularity(
    train_interactions
):
    """
    Recalcule la popularité uniquement avec les données
    disponibles pendant l'entraînement.

    Cela évite d'utiliser l'interaction cachée
    de l'utilisateur cible.
    """

    popularity = (
        train_interactions
        .groupby("resource")["user"]
        .nunique()
        .reset_index(
            name="support_global"
        )
    )

    max_support = (
        popularity["support_global"]
        .max()
    )

    if max_support > 0:

        popularity[
            "popularity_score"
        ] = (
            popularity["support_global"]
            / max_support
        )

    else:

        popularity[
            "popularity_score"
        ] = 0.0

    return popularity


# ======================================================================
# GENERATION DES RECOMMANDATIONS
# ======================================================================

def recommend_for_evaluation(
    target_user,
    train_interactions,
    top_k=TOP_K
):
    """
    Génère des recommandations uniquement
    à partir des informations disponibles
    dans le jeu d'entraînement.
    """

    profiles = build_profiles(
        train_interactions
    )

    if target_user not in profiles:
        return []

    target_resources = profiles[
        target_user
    ]

    # --------------------------------------------------------------
    # VOISINS
    # --------------------------------------------------------------

    neighbors = find_neighbors(
        target_user,
        target_resources,
        profiles
    )

    if neighbors.empty:
        return []

    # --------------------------------------------------------------
    # CANDIDATS
    # --------------------------------------------------------------

    candidate_scores = {}

    for _, neighbor in neighbors.iterrows():

        similar_user = (
            neighbor["similar_user"]
        )

        similarity = float(
            neighbor["similarity"]
        )

        neighbor_resources = profiles[
            similar_user
        ]

        # Recalcul dynamique
        new_resources = (
            neighbor_resources
            - target_resources
        )

        for resource in new_resources:

            if resource not in candidate_scores:

                candidate_scores[
                    resource
                ] = {
                    "neighbor_score": 0.0,
                    "support_users": 0
                }

            candidate_scores[
                resource
            ]["neighbor_score"] += (
                similarity
            )

            candidate_scores[
                resource
            ]["support_users"] += 1

    if not candidate_scores:
        return []

    candidates = pd.DataFrame(
        [
            {
                "resource": resource,
                **values
            }
            for resource, values
            in candidate_scores.items()
        ]
    )

    # --------------------------------------------------------------
    # POPULARITE
    # --------------------------------------------------------------

    popularity = calculate_popularity(
        train_interactions
    )

    candidates = candidates.merge(
        popularity[
            [
                "resource",
                "popularity_score"
            ]
        ],
        on="resource",
        how="left"
    )

    candidates[
        "popularity_score"
    ] = (
        candidates[
            "popularity_score"
        ]
        .fillna(0.0)
        .astype(float)
    )

    # --------------------------------------------------------------
    # NORMALISATION SCORE VOISINS
    # --------------------------------------------------------------

    max_neighbor = (
        candidates[
            "neighbor_score"
        ].max()
    )

    if max_neighbor > 0:

        candidates[
            "neighbor_score_norm"
        ] = (
            candidates[
                "neighbor_score"
            ]
            / max_neighbor
        )

    else:

        candidates[
            "neighbor_score_norm"
        ] = 0.0

    # --------------------------------------------------------------
    # NORMALISATION SUPPORT
    # --------------------------------------------------------------

    max_support = (
        candidates[
            "support_users"
        ].max()
    )

    if max_support > 0:

        candidates[
            "support_score"
        ] = (
            candidates[
                "support_users"
            ]
            / max_support
        )

    else:

        candidates[
            "support_score"
        ] = 0.0

    # --------------------------------------------------------------
    # SCORE FINAL
    # --------------------------------------------------------------

    candidates[
        "final_score"
    ] = (
        NEIGHBOR_WEIGHT
        * candidates[
            "neighbor_score_norm"
        ]
        +
        POPULARITY_WEIGHT
        * candidates[
            "popularity_score"
        ]
        +
        SUPPORT_WEIGHT
        * candidates[
            "support_score"
        ]
    )

    # --------------------------------------------------------------
    # CLASSEMENT
    # --------------------------------------------------------------

    candidates = candidates.sort_values(
        by=[
            "final_score",
            "support_users"
        ],
        ascending=[
            False,
            False
        ]
    )

    return (
        candidates
        .head(top_k)["resource"]
        .tolist()
    )


# ======================================================================
# PRECISION
# ======================================================================

def precision_at_k(
    recommendations,
    relevant
):
    """
    Nombre de recommandations pertinentes
    divisé par K réellement retourné.

    Pour maintenir Precision@10 classique,
    on divise ici par TOP_K.
    """

    if TOP_K == 0:
        return 0.0

    hits = len(
        set(recommendations)
        & set(relevant)
    )

    return hits / TOP_K


# ======================================================================
# RECALL
# ======================================================================

def recall_at_k(
    recommendations,
    relevant
):
    """
    Proportion des ressources cachées retrouvées.
    """

    if not relevant:
        return 0.0

    hits = len(
        set(recommendations)
        & set(relevant)
    )

    return (
        hits
        / len(relevant)
    )


# ======================================================================
# HIT RATE
# ======================================================================

def hit_rate_at_k(
    recommendations,
    relevant
):
    """
    1 si au moins une ressource cachée est retrouvée.
    """

    hit = (
        set(recommendations)
        & set(relevant)
    )

    return (
        1.0
        if hit
        else 0.0
    )


# ======================================================================
# NDCG
# ======================================================================

def ndcg_at_k(
    recommendations,
    relevant
):
    """
    NDCG@K.

    Les recommandations pertinentes placées
    en haut du classement reçoivent davantage de poids.
    """

    if not relevant:
        return 0.0

    relevant = set(
        relevant
    )

    # --------------------------------------------------------------
    # DCG
    # --------------------------------------------------------------

    dcg = 0.0

    for index, resource in enumerate(
        recommendations[:TOP_K]
    ):

        if resource in relevant:

            rank = index + 1

            dcg += (
                1.0
                / math.log2(
                    rank + 1
                )
            )

    # --------------------------------------------------------------
    # IDCG
    # --------------------------------------------------------------

    ideal_hits = min(
        len(relevant),
        TOP_K
    )

    idcg = sum(
        1.0
        / math.log2(
            rank + 1
        )
        for rank in range(
            1,
            ideal_hits + 1
        )
    )

    if idcg == 0:
        return 0.0

    return dcg / idcg


# ======================================================================
# EVALUATION D'UN UTILISATEUR
# ======================================================================

def evaluate_user(
    user,
    interactions,
    rng
):
    """
    Masque TEST_RATIO des ressources de l'utilisateur,
    reconstruit son profil TRAIN puis génère
    les recommandations.
    """

    user_resources = (
        interactions.loc[
            interactions["user"] == user,
            "resource"
        ]
        .drop_duplicates()
        .tolist()
    )

    if len(user_resources) < MIN_INTERACTIONS:
        return None

    # --------------------------------------------------------------
    # NOMBRE D'ELEMENTS DE TEST
    # --------------------------------------------------------------

    test_size = max(
        1,
        int(
            round(
                len(user_resources)
                * TEST_RATIO
            )
        )
    )

    # Garder au moins une ressource dans le train
    test_size = min(
        test_size,
        len(user_resources) - 1
    )

    if test_size <= 0:
        return None

    # --------------------------------------------------------------
    # SELECTION REPRODUCTIBLE
    # --------------------------------------------------------------

    test_resources = set(
        rng.sample(
            user_resources,
            test_size
        )
    )

    # --------------------------------------------------------------
    # CONSTRUCTION DU TRAIN
    # --------------------------------------------------------------

    # On retire uniquement les interactions cachées
    # de l'utilisateur évalué.
    mask = ~(
        (interactions["user"] == user)
        &
        (
            interactions["resource"]
            .isin(test_resources)
        )
    )

    train_interactions = (
        interactions[
            mask
        ]
        .copy()
        .reset_index(drop=True)
    )

    # --------------------------------------------------------------
    # RECOMMANDATIONS
    # --------------------------------------------------------------

    recommendations = (
        recommend_for_evaluation(
            target_user=user,
            train_interactions=train_interactions,
            top_k=TOP_K
        )
    )

    # --------------------------------------------------------------
    # METRIQUES
    # --------------------------------------------------------------

    precision = precision_at_k(
        recommendations,
        test_resources
    )

    recall = recall_at_k(
        recommendations,
        test_resources
    )

    hit_rate = hit_rate_at_k(
        recommendations,
        test_resources
    )

    ndcg = ndcg_at_k(
        recommendations,
        test_resources
    )

    return {
        "user": user,
        "train_resources": (
            len(user_resources)
            - len(test_resources)
        ),
        "test_resources": len(
            test_resources
        ),
        "recommendations": len(
            recommendations
        ),
        "precision": precision,
        "recall": recall,
        "hit_rate": hit_rate,
        "ndcg": ndcg
    }


# ======================================================================
# PROGRAMME PRINCIPAL
# ======================================================================

def main():

    print()
    print("=" * 70)
    print(
        "EVALUATION DU SYSTEME DE RECOMMANDATION V4"
    )
    print("=" * 70)

    interactions = load_interactions()

    print()
    print(
        f"Interactions : "
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

    # --------------------------------------------------------------
    # UTILISATEURS ELIGIBLES
    # --------------------------------------------------------------

    counts = (
        interactions
        .groupby("user")["resource"]
        .nunique()
    )

    eligible_users = (
        counts[
            counts >= MIN_INTERACTIONS
        ]
        .index
        .tolist()
    )

    # --------------------------------------------------------------
    # REPRODUCTIBILITE
    # --------------------------------------------------------------

    rng = random.Random(
        RANDOM_SEED
    )

    # Mélange reproductible
    rng.shuffle(
        eligible_users
    )

    users_to_evaluate = (
        eligible_users[
            :MAX_USERS
        ]
    )

    print()
    print(
        f"Utilisateurs évalués : "
        f"{len(users_to_evaluate)}"
    )

    # --------------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------------

    results = []

    for index, user in enumerate(
        users_to_evaluate,
        start=1
    ):

        result = evaluate_user(
            user,
            interactions,
            rng
        )

        if result is not None:
            results.append(
                result
            )

        # Progression
        if (
            index % 10 == 0
            or index == len(
                users_to_evaluate
            )
        ):

            print(
                f"Progression : "
                f"{index}/"
                f"{len(users_to_evaluate)}"
            )

    if not results:

        print(
            "\nAucun utilisateur "
            "n'a pu être évalué."
        )

        return

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------------
    # MOYENNES
    # --------------------------------------------------------------

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

    # --------------------------------------------------------------
    # DIAGNOSTIC
    # --------------------------------------------------------------

    users_with_recommendations = (
        results_df[
            results_df[
                "recommendations"
            ] > 0
        ]
        .shape[0]
    )

    empty_recommendations = (
        len(results_df)
        - users_with_recommendations
    )

    average_recommendations = (
        results_df[
            "recommendations"
        ].mean()
    )

    # --------------------------------------------------------------
    # AFFICHAGE
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("RESULTATS")
    print("=" * 70)

    print()
    print(
        f"Utilisateurs évalués : "
        f"{len(results_df)}"
    )

    print(
        f"Precision@{TOP_K} : "
        f"{precision:.4f}"
    )

    print(
        f"Recall@{TOP_K}    : "
        f"{recall:.4f}"
    )

    print(
        f"Hit Rate@{TOP_K}  : "
        f"{hit_rate:.4f}"
    )

    print(
        f"NDCG@{TOP_K}      : "
        f"{ndcg:.4f}"
    )

    # --------------------------------------------------------------
    # INFORMATIONS COMPLEMENTAIRES
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("DIAGNOSTIC")
    print("=" * 70)

    print()
    print(
        "Utilisateurs avec recommandations : "
        f"{users_with_recommendations}"
    )

    print(
        "Utilisateurs sans recommandation : "
        f"{empty_recommendations}"
    )

    print(
        "Nombre moyen de recommandations : "
        f"{average_recommendations:.2f}"
    )

    # --------------------------------------------------------------
    # SAUVEGARDE DETAILLEE
    # --------------------------------------------------------------

    output_file = (
        PROCESSED_DIR
        / "evaluation_results_v4.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print()
    print(
        f"✓ Détails sauvegardés : "
        f"{output_file}"
    )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()