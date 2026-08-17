"""
======================================================================
SHAREPOINT RECOMMENDER - RECOMMENDER V4
======================================================================

Moteur de recommandation hybride.

Sources utilisées :
    1. Similarité entre utilisateurs
    2. Ressources réellement nouvelles
    3. Popularité des ressources
    4. Contexte hiérarchique SharePoint

Objectif :
    Ne pas recommander uniquement les ressources déjà très populaires
    ou déjà connues par l'utilisateur.

======================================================================
"""

from pathlib import Path

import pandas as pd


# ======================================================================
# CONFIGURATION
# ======================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "processed"

INTERACTIONS_FILE = PROCESSED_DIR / "interactions.parquet"
RESOURCES_FILE = PROCESSED_DIR / "resources.parquet"
NEIGHBORS_FILE = PROCESSED_DIR / "user_neighbors.parquet"
POPULARITY_FILE = PROCESSED_DIR / "resource_popularity.parquet"


# Nombre maximum de voisins utilisés
TOP_NEIGHBORS = 30

# Nombre de recommandations finales
TOP_RECOMMENDATIONS = 10


# ======================================================================
# CHARGEMENT DES DONNÉES
# ======================================================================

def load_data():
    """
    Charge les différents fichiers générés par le preprocessing.
    """

    interactions = pd.read_parquet(INTERACTIONS_FILE)
    resources = pd.read_parquet(RESOURCES_FILE)
    neighbors = pd.read_parquet(NEIGHBORS_FILE)
    popularity = pd.read_parquet(POPULARITY_FILE)

    return interactions, resources, neighbors, popularity


# ======================================================================
# RESSOURCES DÉJÀ CONNUES
# ======================================================================

def get_user_resources(interactions, user):
    """
    Retourne l'ensemble des ressources déjà connues par l'utilisateur.
    """

    return set(
        interactions.loc[
            interactions["user"] == user,
            "resource"
        ]
    )


# ======================================================================
# VOISINS PERTINENTS
# ======================================================================

def get_relevant_neighbors(neighbors, user):
    """
    Sélectionne les voisins les plus pertinents.

    On privilégie :
        - une bonne similarité
        - un bon Jaccard
        - surtout la présence de nouvelles ressources

    Les voisins qui n'apportent aucune nouvelle ressource
    sont écartés.
    """

    result = neighbors[
        (neighbors["user"] == user) &
        (neighbors["new_resources"] > 0)
    ].copy()

    if result.empty:
        return result

    # Score combiné de proximité
    result["neighbor_score"] = (
        0.50 * result["similarity"]
        + 0.30 * result["jaccard"]
        + 0.20 * result["containment"]
    )

    result = result.sort_values(
        "neighbor_score",
        ascending=False
    )

    return result.head(TOP_NEIGHBORS)


# ======================================================================
# GÉNÉRATION DES CANDIDATS
# ======================================================================

def generate_candidates(
    interactions,
    neighbors,
    user
):
    """
    Construit la liste des ressources candidates.

    Une ressource devient candidate lorsqu'elle est utilisée
    par un voisin mais pas encore utilisée par l'utilisateur cible.
    """

    user_resources = get_user_resources(
        interactions,
        user
    )

    relevant_neighbors = get_relevant_neighbors(
        neighbors,
        user
    )

    if relevant_neighbors.empty:
        return pd.DataFrame()

    candidates = {}

    for _, neighbor in relevant_neighbors.iterrows():

        similar_user = neighbor["similar_user"]
        neighbor_score = neighbor["neighbor_score"]

        neighbor_resources = set(
            interactions.loc[
                interactions["user"] == similar_user,
                "resource"
            ]
        )

        # Ressources réellement nouvelles
        new_resources = neighbor_resources - user_resources

        for resource in new_resources:

            if resource not in candidates:
                candidates[resource] = {
                    "resource": resource,
                    "neighbor_score": 0.0,
                    "support_users": 0,
                }

            candidates[resource]["neighbor_score"] += (
                neighbor_score
            )

            candidates[resource]["support_users"] += 1

    if not candidates:
        return pd.DataFrame()

    return pd.DataFrame(
        candidates.values()
    )


# ======================================================================
# SCORE DE POPULARITÉ
# ======================================================================

def add_popularity_score(
    candidates,
    popularity
):
    """
    Ajoute le score de popularité aux candidats.

    La popularité est utilisée comme signal secondaire.
    Elle ne doit pas dominer la similarité utilisateur.
    """

    if candidates.empty:
        return candidates

    candidates = candidates.merge(
        popularity[
            [
                "resource",
                "support_users",
                "popularity_score"
            ]
        ],
        on="resource",
        how="left",
        suffixes=("", "_popularity")
    )

    # Si une ressource n'existe pas dans le fichier de popularité
    candidates["popularity_score"] = (
        candidates["popularity_score"]
        .fillna(0)
    )

    return candidates


# ======================================================================
# SCORE FINAL
# ======================================================================

def calculate_final_score(candidates):
    """
    Calcule le score final de recommandation.

    Pondération :

        70 % proximité utilisateurs
        20 % popularité
        10 % diversité/support

    L'objectif est de privilégier les ressources pertinentes
    plutôt que simplement les ressources populaires.
    """

    if candidates.empty:
        return candidates

    # Normalisation du score voisin
    max_neighbor_score = candidates[
        "neighbor_score"
    ].max()

    if max_neighbor_score > 0:
        candidates["neighbor_score_norm"] = (
            candidates["neighbor_score"]
            / max_neighbor_score
        )
    else:
        candidates["neighbor_score_norm"] = 0

    # Score de diversité :
    # plusieurs voisins indépendants recommandant la même
    # ressource augmentent sa fiabilité.
    max_support = candidates[
        "support_users"
    ].max()

    if max_support > 0:
        candidates["support_score"] = (
            candidates["support_users"]
            / max_support
        )
    else:
        candidates["support_score"] = 0

    # Score final
    candidates["final_score"] = (
        0.70 * candidates["neighbor_score_norm"]
        + 0.20 * candidates["popularity_score"]
        + 0.10 * candidates["support_score"]
    )

    return candidates


# ======================================================================
# AJOUT DES INFORMATIONS SHAREPOINT
# ======================================================================

def add_resource_information(
    candidates,
    resources
):
    """
    Ajoute les informations détaillées de la ressource :

        site
        sous-site
        bibliothèque
        liste
    """

    if candidates.empty:
        return candidates

    columns = [
        "resource",
        "site",
        "sous-site",
        "bibliothèque",
        "liste"
    ]

    candidates = candidates.merge(
        resources[columns],
        on="resource",
        how="left"
    )

    return candidates


# ======================================================================
# RECOMMANDATION
# ======================================================================

def recommend(
    user,
    interactions,
    resources,
    neighbors,
    popularity,
    top_n=TOP_RECOMMENDATIONS
):
    """
    Fonction principale du système de recommandation.
    """

    candidates = generate_candidates(
        interactions,
        neighbors,
        user
    )

    # Aucun candidat
    if candidates.empty:
        return pd.DataFrame()

    candidates = add_popularity_score(
        candidates,
        popularity
    )

    candidates = calculate_final_score(
        candidates
    )

    candidates = add_resource_information(
        candidates,
        resources
    )

    # Tri final
    candidates = candidates.sort_values(
        [
            "final_score",
            "support_users"
        ],
        ascending=False
    )

    # Sélection finale
    recommendations = candidates.head(top_n).copy()

    # Pourcentage lisible
    recommendations["score_percent"] = (
        recommendations["final_score"] * 100
    ).round(2)

    # Type de recommandation
    recommendations["recommendation_type"] = (
        "Collaborative filtering"
    )

    # Explication
    recommendations["explanation"] = (
        recommendations.apply(
            lambda row:
            f"Ressource utilisée par "
            f"{int(row['support_users'])} utilisateur(s) "
            f"similaire(s)",
            axis=1
        )
    )

    return recommendations


# ======================================================================
# AFFICHAGE
# ======================================================================

def display_recommendations(
    recommendations,
    user
):
    """
    Affiche les recommandations dans un format lisible.
    """

    print()
    print("=" * 70)
    print(f"RECOMMANDATIONS POUR : {user}")
    print("=" * 70)

    if recommendations.empty:

        print()
        print("Aucune nouvelle recommandation disponible.")
        print()

        return

    columns = [
        "resource",
        "site",
        "sous-site",
        "bibliothèque",
        "liste",
        "score_percent",
        "support_users",
        "recommendation_type",
        "explanation"
    ]

    print(
        recommendations[columns]
        .to_string(index=False)
    )

    print()
    print("=" * 70)


# ======================================================================
# PROGRAMME PRINCIPAL
# ======================================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("SHAREPOINT RECOMMENDER - RECOMMENDER V4")
    print("=" * 70)

    # --------------------------------------------------------------
    # Chargement
    # --------------------------------------------------------------

    interactions, resources, neighbors, popularity = load_data()

    print()
    print(f"Interactions : {len(interactions):,}")
    print(
        f"Utilisateurs : "
        f"{interactions['user'].nunique():,}"
    )
    print(
        f"Ressources : "
        f"{interactions['resource'].nunique():,}"
    )

    # --------------------------------------------------------------
    # Utilisateur de test
    # --------------------------------------------------------------

    user = "fabien"

    print()
    print(f"Utilisateur : {user}")

    # --------------------------------------------------------------
    # Recommandation
    # --------------------------------------------------------------

    recommendations = recommend(
        user=user,
        interactions=interactions,
        resources=resources,
        neighbors=neighbors,
        popularity=popularity,
        top_n=TOP_RECOMMENDATIONS
    )

    # --------------------------------------------------------------
    # Affichage
    # --------------------------------------------------------------

    display_recommendations(
        recommendations,
        user
    )