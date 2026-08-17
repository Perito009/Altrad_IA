"""
======================================================================
SHAREPOINT RECOMMENDER - SIMILARITY V4
======================================================================

Calcul de similarité entre utilisateurs SharePoint.

Objectif :
    Identifier les utilisateurs ayant des profils SharePoint proches
    ET identifier ceux qui peuvent apporter de nouvelles ressources.

Fichier d'entrée :
    processed/interactions.parquet

Colonnes attendues :
    user
    resource

Fichier de sortie :
    processed/user_neighbors.parquet

Colonnes produites :

    user
    similar_user
    similarity
    jaccard
    containment
    common_resources
    user_resources
    neighbor_resources
    new_resources

Principe :

    1. Construire le profil de chaque utilisateur
    2. Comparer les profils
    3. Calculer Jaccard
    4. Calculer Containment
    5. Calculer le nombre de ressources nouvelles
    6. Calculer un score final
    7. Conserver les meilleurs voisins

======================================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ======================================================================
# CONFIGURATION
# ======================================================================

# Répertoire contenant ce fichier
BASE_DIR = Path(__file__).resolve().parent

# Répertoire racine du projet
PROJECT_DIR = BASE_DIR.parent

# Répertoire contenant les données préparées
PROCESSED_DIR = PROJECT_DIR / "processed"

# Fichier des interactions
INTERACTIONS_FILE = PROCESSED_DIR / "interactions.parquet"

# Fichier de sortie
OUTPUT_FILE = PROCESSED_DIR / "user_neighbors.parquet"


# ======================================================================
# PARAMÈTRES DU MODÈLE
# ======================================================================

# Nombre maximum de voisins conservés par utilisateur.
TOP_K_NEIGHBORS = 20

# Nombre minimum de ressources communes pour considérer
# deux utilisateurs comme suffisamment proches.
MIN_COMMON_RESOURCES = 2

# Score minimum de similarité.
MIN_SIMILARITY = 0.05


# ======================================================================
# CHARGEMENT DES DONNÉES
# ======================================================================

def load_interactions():
    """
    Charge les interactions depuis interactions.parquet.

    Retourne
    --------
    pandas.DataFrame
        DataFrame contenant au minimum :
            user
            resource
    """

    if not INTERACTIONS_FILE.exists():

        raise FileNotFoundError(
            "\nFichier interactions.parquet introuvable :\n"
            f"{INTERACTIONS_FILE}\n\n"
            "Lance d'abord :\n"
            "python SharePoint-Recommender/preprocessing.py"
        )

    interactions = pd.read_parquet(
        INTERACTIONS_FILE
    )

    # Vérification de la structure
    required_columns = {
        "user",
        "resource",
    }

    missing_columns = (
        required_columns
        - set(interactions.columns)
    )

    if missing_columns:

        raise ValueError(
            "Colonnes manquantes dans interactions.parquet : "
            f"{sorted(missing_columns)}"
        )

    # Suppression des valeurs manquantes
    interactions = interactions.dropna(
        subset=[
            "user",
            "resource",
        ]
    ).copy()

    # Conversion en chaînes
    interactions["user"] = (
        interactions["user"]
        .astype(str)
        .str.strip()
    )

    interactions["resource"] = (
        interactions["resource"]
        .astype(str)
        .str.strip()
    )

    # Suppression des doublons
    interactions = (
        interactions
        .drop_duplicates(
            subset=[
                "user",
                "resource",
            ]
        )
        .reset_index(drop=True)
    )

    return interactions


# ======================================================================
# CONSTRUCTION DES PROFILS UTILISATEURS
# ======================================================================

def build_user_profiles(interactions):
    """
    Construit un profil sous forme d'ensemble de ressources
    pour chaque utilisateur.

    Exemple :

        fabien -> {
            ressource_A,
            ressource_B,
            ressource_C
        }

    Retourne
    --------
    dict
        {
            utilisateur: set(ressources)
        }
    """

    profiles = {}

    for user, group in interactions.groupby("user"):

        profiles[user] = set(
            group["resource"]
        )

    return profiles


# ======================================================================
# CALCUL DE JACCARD
# ======================================================================

def calculate_jaccard(
    user_resources,
    neighbor_resources,
):
    """
    Calcule la similarité de Jaccard.

    Formule :

        J(A,B) = |A ∩ B| / |A ∪ B|

    Valeur comprise entre 0 et 1.

    1.0 = profils identiques
    0.0 = aucune ressource commune
    """

    intersection = (
        user_resources
        & neighbor_resources
    )

    union = (
        user_resources
        | neighbor_resources
    )

    if not union:

        return 0.0

    return (
        len(intersection)
        / len(union)
    )


# ======================================================================
# CALCUL DU CONTAINMENT
# ======================================================================

def calculate_containment(
    user_resources,
    neighbor_resources,
):
    """
    Calcule le niveau de containment entre deux profils.

    Le containment répond à la question :

        "Quelle proportion du plus petit profil
         est contenue dans le plus grand ?"

    Formule :

        |A ∩ B| / min(|A|, |B|)

    Exemple :

        Fabien = 161 ressources
        Hélène = 38 ressources
        Communes = 38

        containment = 38 / 38 = 1.0

    Cela signifie que toutes les ressources d'Hélène
    sont présentes chez Fabien.
    """

    if (
        not user_resources
        or not neighbor_resources
    ):

        return 0.0

    intersection = (
        user_resources
        & neighbor_resources
    )

    minimum_size = min(
        len(user_resources),
        len(neighbor_resources),
    )

    if minimum_size == 0:

        return 0.0

    return (
        len(intersection)
        / minimum_size
    )


# ======================================================================
# CALCUL D'UNE PAIRE D'UTILISATEURS
# ======================================================================

def calculate_pair_metrics(
    user_resources,
    neighbor_resources,
):
    """
    Calcule toutes les métriques entre deux utilisateurs.

    Retourne
    --------
    dict
        Métriques de similarité.
    """

    common_resources = (
        user_resources
        & neighbor_resources
    )

    new_resources = (
        neighbor_resources
        - user_resources
    )

    jaccard = calculate_jaccard(
        user_resources,
        neighbor_resources,
    )

    containment = calculate_containment(
        user_resources,
        neighbor_resources,
    )

    return {
        "jaccard": jaccard,
        "containment": containment,
        "common_resources": len(
            common_resources
        ),
        "user_resources": len(
            user_resources
        ),
        "neighbor_resources": len(
            neighbor_resources
        ),
        "new_resources": len(
            new_resources
        ),
    }


# ======================================================================
# SCORE FINAL DE SIMILARITÉ
# ======================================================================

def calculate_similarity(
    jaccard,
    containment,
):
    """
    Calcule le score final de similarité.

    Nous combinons :

        60 % Jaccard
        40 % Containment

    Pourquoi ?

    Jaccard mesure la similarité globale.

    Containment permet de mieux identifier les profils
    où l'un des utilisateurs constitue un sous-ensemble
    très proche de l'autre.

    Exemple Fabien / Hélène :

        Jaccard     = 0.236
        Containment = 1.000

    Score :

        0.60 * 0.236
        +
        0.40 * 1.000

        = 0.5416
    """

    return (
        0.60 * jaccard
        +
        0.40 * containment
    )


# ======================================================================
# CALCUL DES VOISINS
# ======================================================================

def calculate_neighbors(
    profiles,
    top_k=TOP_K_NEIGHBORS,
):
    """
    Calcule les meilleurs voisins pour chaque utilisateur.

    Pour chaque paire :

        - Jaccard
        - Containment
        - ressources communes
        - ressources nouvelles
        - score final

    sont calculés.

    Retourne
    --------
    pandas.DataFrame
    """

    users = list(
        profiles.keys()
    )

    results = []

    total_users = len(users)

    print(
        f"\nCalcul des similarités pour "
        f"{total_users:,} utilisateurs..."
    )

    # ==============================================================
    # COMPARAISON DES UTILISATEURS
    # ==============================================================

    for index, user in enumerate(users):

        user_resources = profiles[user]

        user_results = []

        for similar_user in users:

            # ------------------------------------------------------
            # Un utilisateur ne peut pas être son propre voisin.
            # ------------------------------------------------------

            if user == similar_user:
                continue

            neighbor_resources = (
                profiles[similar_user]
            )

            # ------------------------------------------------------
            # Calcul des métriques
            # ------------------------------------------------------

            metrics = calculate_pair_metrics(
                user_resources,
                neighbor_resources,
            )

            # ------------------------------------------------------
            # Filtre sur le nombre de ressources communes
            # ------------------------------------------------------

            if (
                metrics["common_resources"]
                < MIN_COMMON_RESOURCES
            ):
                continue

            # ------------------------------------------------------
            # Score final
            # ------------------------------------------------------

            similarity = calculate_similarity(
                metrics["jaccard"],
                metrics["containment"],
            )

            if similarity < MIN_SIMILARITY:
                continue

            # ------------------------------------------------------
            # Enregistrement
            # ------------------------------------------------------

            user_results.append(
                {
                    "user": user,
                    "similar_user": similar_user,
                    "similarity": similarity,
                    "jaccard": metrics["jaccard"],
                    "containment": metrics["containment"],
                    "common_resources": metrics[
                        "common_resources"
                    ],
                    "user_resources": metrics[
                        "user_resources"
                    ],
                    "neighbor_resources": metrics[
                        "neighbor_resources"
                    ],
                    "new_resources": metrics[
                        "new_resources"
                    ],
                }
            )

        # ==========================================================
        # TRI DES VOISINS
        # ==========================================================

        if user_results:

            user_df = pd.DataFrame(
                user_results
            )

            # ------------------------------------------------------
            # On privilégie d'abord la similarité.
            #
            # En cas d'égalité, les utilisateurs apportant
            # davantage de ressources nouvelles passent devant.
            # ------------------------------------------------------

            user_df = user_df.sort_values(
                by=[
                    "similarity",
                    "new_resources",
                    "common_resources",
                ],
                ascending=[
                    False,
                    False,
                    False,
                ],
            )

            # ------------------------------------------------------
            # Conservation des TOP K
            # ------------------------------------------------------

            user_df = user_df.head(
                top_k
            )

            results.append(
                user_df
            )

        # ==========================================================
        # PROGRESSION
        # ==========================================================

        if (
            (index + 1) % 50 == 0
            or index == 0
            or index + 1 == total_users
        ):

            print(
                f"  Progression : "
                f"{index + 1:,}/{total_users:,}"
            )

    # ==================================================================
    # ASSEMBLAGE
    # ==================================================================

    if not results:

        return pd.DataFrame(
            columns=[
                "user",
                "similar_user",
                "similarity",
                "jaccard",
                "containment",
                "common_resources",
                "user_resources",
                "neighbor_resources",
                "new_resources",
            ]
        )

    neighbors = pd.concat(
        results,
        ignore_index=True,
    )

    return neighbors


# ======================================================================
# SAUVEGARDE
# ======================================================================

def save_neighbors(neighbors):
    """
    Sauvegarde les voisins dans user_neighbors.parquet.
    """

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    neighbors.to_parquet(
        OUTPUT_FILE,
        index=False,
    )


# ======================================================================
# STATISTIQUES
# ======================================================================

def display_statistics(
    interactions,
    profiles,
    neighbors,
):
    """
    Affiche les statistiques du calcul.
    """

    print()
    print("=" * 70)
    print("STATISTIQUES")
    print("=" * 70)

    print(
        f"Utilisateurs       : "
        f"{len(profiles):,}"
    )

    print(
        f"Interactions       : "
        f"{len(interactions):,}"
    )

    print(
        f"Paires conservées  : "
        f"{len(neighbors):,}"
    )

    if not neighbors.empty:

        print(
            f"Similarité moyenne : "
            f"{neighbors['similarity'].mean():.4f}"
        )

        print(
            f"Jaccard moyen      : "
            f"{neighbors['jaccard'].mean():.4f}"
        )

        print(
            f"Containment moyen  : "
            f"{neighbors['containment'].mean():.4f}"
        )

        print(
            f"Nouvelles ressources "
            f"moyennes : "
            f"{neighbors['new_resources'].mean():.2f}"
        )

        users_with_new_resources = (
            neighbors[
                neighbors["new_resources"] > 0
            ]["user"]
            .nunique()
        )

        print(
            f"Utilisateurs ayant "
            f"des voisins avec nouvelles ressources : "
            f"{users_with_new_resources:,}"
        )


# ======================================================================
# EXEMPLE FABien
# ======================================================================

def display_example(
    neighbors,
    user="fabien",
):
    """
    Affiche un exemple de voisins pour contrôler le résultat.

    Cette partie est uniquement destinée au diagnostic.
    """

    if neighbors.empty:
        return

    example = neighbors[
        neighbors["user"] == user
    ].copy()

    if example.empty:

        print(
            f"\nAucun voisin trouvé pour '{user}'."
        )

        return

    print()
    print("=" * 70)
    print(
        f"EXEMPLE - VOISINS DE {user.upper()}"
    )
    print("=" * 70)

    print(
        example[
            [
                "user",
                "similar_user",
                "similarity",
                "jaccard",
                "containment",
                "common_resources",
                "user_resources",
                "neighbor_resources",
                "new_resources",
            ]
        ].to_string(
            index=False
        )
    )


# ======================================================================
# PROGRAMME PRINCIPAL
# ======================================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("SHAREPOINT RECOMMENDER - SIMILARITY V4")
    print("=" * 70)

    # ==============================================================
    # CHARGEMENT
    # ==============================================================

    print(
        "\nChargement des interactions..."
    )

    interactions = load_interactions()

    print(
        f"✓ Interactions chargées : "
        f"{len(interactions):,}"
    )

    # ==============================================================
    # PROFILS
    # ==============================================================

    print(
        "\nConstruction des profils utilisateurs..."
    )

    profiles = build_user_profiles(
        interactions
    )

    print(
        f"✓ Utilisateurs : "
        f"{len(profiles):,}"
    )

    # ==============================================================
    # CALCUL
    # ==============================================================

    neighbors = calculate_neighbors(
        profiles,
        top_k=TOP_K_NEIGHBORS,
    )

    # ==============================================================
    # SAUVEGARDE
    # ==============================================================

    save_neighbors(
        neighbors
    )

    print()
    print(
        f"✓ Fichier créé : "
        f"{OUTPUT_FILE}"
    )

    # ==============================================================
    # STATISTIQUES
    # ==============================================================

    display_statistics(
        interactions,
        profiles,
        neighbors,
    )

    # ==============================================================
    # EXEMPLE
    # ==============================================================

    display_example(
        neighbors,
        user="fabien",
    )

    print()
    print("=" * 70)
    print("CALCUL TERMINÉ")
    print("=" * 70)