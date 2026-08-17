import streamlit as st
import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Altrad_IA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CHEMINS DES FICHIERS
# ============================================================



# Le dossier processed se trouve à la racine du projet
PROCESSED_DIR = Path("processed")

INTERACTIONS_FILE = PROCESSED_DIR / "interactions.parquet"
RESOURCES_FILE = PROCESSED_DIR / "resources.parquet"
NEIGHBORS_FILE = PROCESSED_DIR / "user_neighbors.parquet"
POPULARITY_FILE = PROCESSED_DIR / "resource_popularity.parquet"


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 18px;
        color: #666;
        margin-top: 0;
        margin-bottom: 30px;
    }

    .section-title {
        font-size: 26px;
        font-weight: 600;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CHARGEMENT DES DONNÉES
# ============================================================

@st.cache_data
def load_data():
    """
    Charge les différents fichiers générés par le pipeline ML.

    Le cache permet d'éviter de relire les fichiers Parquet
    à chaque interaction avec l'application.
    """

    interactions = pd.read_parquet(INTERACTIONS_FILE)

    resources = pd.read_parquet(RESOURCES_FILE)

    neighbors = pd.read_parquet(NEIGHBORS_FILE)

    popularity = pd.read_parquet(POPULARITY_FILE)

    return interactions, resources, neighbors, popularity


# ============================================================
# VERIFICATION DES FICHIERS
# ============================================================

missing_files = []

for file in [
    INTERACTIONS_FILE,
    RESOURCES_FILE,
    NEIGHBORS_FILE,
    POPULARITY_FILE
]:

    if not file.exists():
        missing_files.append(str(file))


if missing_files:

    st.error("Certains fichiers nécessaires sont absents.")

    st.write("Fichiers manquants :")

    for file in missing_files:
        st.code(file)

    st.info(
        "Exécute d'abord preprocessing.py, similarity.py et popularity.py."
    )

    st.stop()


# ============================================================
# CHARGEMENT
# ============================================================

interactions, resources, neighbors, popularity = load_data()


# ============================================================
# VERIFICATION DES COLONNES
# ============================================================

required_interaction_columns = {
    "user",
    "resource"
}

required_resource_columns = {
    "resource",
    "site",
    "sous-site",
    "bibliothèque",
    "liste"
}

required_neighbor_columns = {
    "user",
    "similar_user",
    "similarity"
}

required_popularity_columns = {
    "resource",
    "support_users",
    "popularity_score"
}


if not required_interaction_columns.issubset(interactions.columns):

    st.error(
        "Le fichier interactions.parquet ne possède pas "
        "les colonnes attendues."
    )

    st.write(interactions.columns.tolist())

    st.stop()


if not required_resource_columns.issubset(resources.columns):

    st.error(
        "Le fichier resources.parquet ne possède pas "
        "les colonnes attendues."
    )

    st.write(resources.columns.tolist())

    st.stop()


if not required_neighbor_columns.issubset(neighbors.columns):

    st.error(
        "Le fichier user_neighbors.parquet ne possède pas "
        "les colonnes attendues."
    )

    st.write(neighbors.columns.tolist())

    st.stop()


# ============================================================
# TITRE GLOBAL
# ============================================================

st.markdown(
    '<div class="main-title">Bienvenue sur Altrad_IA</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    "Analyse des utilisateurs et système de recommandation SharePoint"
    "</div>",
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choisir une page",
    [
        "👥 Utilisateurs",
        "🤖 Recommandations intelligentes"
    ]
)


# ============================================================
# PAGE 1 : UTILISATEURS
# ============================================================

if page == "👥 Utilisateurs":

    st.markdown(
        '<div class="section-title">'
        "Utilisateurs et ressources"
        "</div>",
        unsafe_allow_html=True
    )

    st.write(
        "Cette page permet de consulter simplement les ressources "
        "associées à chaque utilisateur."
    )

    # --------------------------------------------------------
    # STATISTIQUES
    # --------------------------------------------------------

    nb_users = interactions["user"].nunique()

    nb_resources = interactions["resource"].nunique()

    nb_interactions = len(interactions)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Utilisateurs",
            f"{nb_users:,}".replace(",", " ")
        )

    with col2:
        st.metric(
            "Ressources",
            f"{nb_resources:,}".replace(",", " ")
        )

    with col3:
        st.metric(
            "Interactions",
            f"{nb_interactions:,}".replace(",", " ")
        )

    st.divider()

    # --------------------------------------------------------
    # SELECTION UTILISATEUR
    # --------------------------------------------------------

    users = sorted(
        interactions["user"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_user = st.selectbox(
        "Sélectionner un utilisateur",
        users
    )

    # --------------------------------------------------------
    # RESSOURCES DE L'UTILISATEUR
    # --------------------------------------------------------

    user_interactions = interactions[
        interactions["user"] == selected_user
    ].copy()

    user_resources = user_interactions.merge(
        resources,
        on="resource",
        how="left"
    )

    st.subheader(
        f"Ressources utilisées par : {selected_user}"
    )

    st.metric(
        "Nombre de ressources",
        len(user_resources)
    )

    # --------------------------------------------------------
    # FILTRE
    # --------------------------------------------------------

    search = st.text_input(
        "🔎 Rechercher une ressource",
        placeholder="Nom du site, sous-site, bibliothèque..."
    )

    if search:

        search = search.lower()

        mask = (
            user_resources["resource"]
            .fillna("")
            .str.lower()
            .str.contains(search, na=False)
        )

        user_resources = user_resources[mask]

    # --------------------------------------------------------
    # TABLEAU
    # --------------------------------------------------------

    display_columns = [
        "resource",
        "site",
        "sous-site",
        "bibliothèque",
        "liste"
    ]

    st.dataframe(
        user_resources[display_columns],
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PAGE 2 : RECOMMANDATIONS
# ============================================================

elif page == "🤖 Recommandations intelligentes":

    st.markdown(
        '<div class="section-title">'
        "Système de recommandations intelligentes"
        "</div>",
        unsafe_allow_html=True
    )

    st.write(
        "Le système cherche des ressources qu'un utilisateur "
        "pourrait trouver pertinentes à partir de son profil, "
        "de ses utilisateurs similaires et de la popularité "
        "des ressources."
    )

    # --------------------------------------------------------
    # SELECTION UTILISATEUR
    # --------------------------------------------------------

    users = sorted(
        interactions["user"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_user = st.selectbox(
        "👤 Utilisateur à recommander",
        users,
        key="recommendation_user"
    )

    # --------------------------------------------------------
    # PROFIL UTILISATEUR
    # --------------------------------------------------------

    user_resources = set(
        interactions[
            interactions["user"] == selected_user
        ]["resource"]
    )

    user_count = len(user_resources)

    # --------------------------------------------------------
    # VOISINS
    # --------------------------------------------------------

    user_neighbors = neighbors[
        neighbors["user"] == selected_user
    ].copy()

    user_neighbors = user_neighbors.sort_values(
        "similarity",
        ascending=False
    )

    # --------------------------------------------------------
    # NOUVELLES RESSOURCES
    # --------------------------------------------------------

    # On va chercher les ressources utilisées par les voisins
    # mais pas encore utilisées par l'utilisateur sélectionné.

    candidate_scores = {}

    candidate_support = {}

    candidate_neighbors = {}

    for _, row in user_neighbors.iterrows():

        similar_user = row["similar_user"]

        similarity = float(row["similarity"])

        neighbor_resources = set(
            interactions[
                interactions["user"] == similar_user
            ]["resource"]
        )

        new_resources = (
            neighbor_resources - user_resources
        )

        for resource in new_resources:

            # Contribution du voisin au score
            contribution = similarity

            candidate_scores[resource] = (
                candidate_scores.get(resource, 0)
                + contribution
            )

            candidate_support[resource] = (
                candidate_support.get(resource, 0)
                + 1
            )

            candidate_neighbors.setdefault(
                resource,
                []
            ).append(similar_user)

    # --------------------------------------------------------
    # POPULARITE
    # --------------------------------------------------------

    popularity_dict = {}

    if not popularity.empty:

        for _, row in popularity.iterrows():

            popularity_dict[
                row["resource"]
            ] = float(
                row["popularity_score"]
            )

    # --------------------------------------------------------
    # CONSTRUCTION DES RECOMMANDATIONS
    # --------------------------------------------------------

    recommendations = []

    for resource, collaborative_score in candidate_scores.items():

        # Ressource déjà utilisée :
        # elle ne doit pas être recommandée.
        if resource in user_resources:
            continue

        support_users = candidate_support.get(
            resource,
            0
        )

        popularity_score = popularity_dict.get(
            resource,
            0
        )

        # ----------------------------------------------------
        # SCORE FINAL
        # ----------------------------------------------------
        #
        # 70 % : similarité collaborative
        # 30 % : popularité
        #
        # Cela permet de ne pas uniquement recommander
        # les ressources les plus populaires.
        # ----------------------------------------------------

        final_score = (
            0.70 * collaborative_score
            + 0.30 * popularity_score
        )

        recommendations.append(
            {
                "resource": resource,
                "collaborative_score": collaborative_score,
                "popularity_score": popularity_score,
                "support_users": support_users,
                "final_score": final_score,
                "neighbors": candidate_neighbors.get(
                    resource,
                    []
                )
            }
        )

    recommendations_df = pd.DataFrame(
        recommendations
    )

    # --------------------------------------------------------
    # INFORMATIONS DU PROFIL
    # --------------------------------------------------------

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Ressources utilisées",
            user_count
        )

    with col2:

        st.metric(
            "Voisins similaires",
            len(user_neighbors)
        )

    with col3:

        st.metric(
            "Ressources candidates",
            len(recommendations_df)
        )

    # --------------------------------------------------------
    # VOISINS
    # --------------------------------------------------------

    st.subheader(
        "👥 Utilisateurs similaires"
    )

    if user_neighbors.empty:

        st.warning(
            "Aucun utilisateur similaire disponible."
        )

    else:

        neighbors_display = user_neighbors.head(10).copy()

        neighbors_display[
            "similarity_percent"
        ] = (
            neighbors_display["similarity"] * 100
        ).round(2)

        neighbors_display = neighbors_display[
            [
                "similar_user",
                "similarity_percent"
            ]
        ]

        neighbors_display.columns = [
            "Utilisateur similaire",
            "Similarité (%)"
        ]

        st.dataframe(
            neighbors_display,
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------------
    # RECOMMANDATIONS
    # --------------------------------------------------------

    st.subheader(
        f"🎯 Recommandations pour {selected_user}"
    )

    if recommendations_df.empty:

        st.info(
            "Aucune nouvelle recommandation disponible "
            "avec les données actuelles."
        )

        st.write(
            "Cela signifie généralement que les utilisateurs "
            "similaires utilisent déjà les mêmes ressources "
            "que l'utilisateur sélectionné."
        )

    else:

        # Tri par score final
        recommendations_df = recommendations_df.sort_values(
            "final_score",
            ascending=False
        ).head(10)

        # ----------------------------------------------------
        # AJOUT DES INFORMATIONS DE RESSOURCE
        # ----------------------------------------------------

        recommendations_df = recommendations_df.merge(
            resources,
            on="resource",
            how="left"
        )

        # ----------------------------------------------------
        # SCORE EN POURCENTAGE
        # ----------------------------------------------------

        recommendations_df[
            "score_percent"
        ] = (
            recommendations_df["final_score"] * 100
        ).round(2)

        # ----------------------------------------------------
        # TYPE DE RECOMMANDATION
        # ----------------------------------------------------

        recommendations_df[
            "recommendation_type"
        ] = "Collaborative + Popularité"

        # ----------------------------------------------------
        # EXPLICATION
        # ----------------------------------------------------

        recommendations_df[
            "explanation"
        ] = (
            "Ressource utilisée par des utilisateurs "
            "similaires et renforcée par sa popularité."
        )

        # ----------------------------------------------------
        # TABLEAU FINAL
        # ----------------------------------------------------

        display_columns = [
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

        st.dataframe(
            recommendations_df[display_columns],
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # DETAIL D'UNE RECOMMANDATION
        # ----------------------------------------------------

        st.subheader(
            "🔎 Explication d'une recommandation"
        )

        selected_resource = st.selectbox(
            "Choisir une ressource",
            recommendations_df["resource"].tolist()
        )

        selected_row = recommendations_df[
            recommendations_df["resource"]
            == selected_resource
        ].iloc[0]

        st.write(
            f"**Ressource :** {selected_resource}"
        )

        st.write(
            f"**Score final :** "
            f"{selected_row['score_percent']:.2f} %"
        )

        st.write(
            f"**Utilisateurs similaires qui la possèdent :** "
            f"{int(selected_row['support_users'])}"
        )

        st.write(
            f"**Popularité :** "
            f"{selected_row['popularity_score']:.3f}"
        )

        st.info(
            selected_row["explanation"]
        )

    # --------------------------------------------------------
    # EXPLICATION DU MODELE
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "🧠 Comment fonctionne la recommandation ?"
    )

    st.markdown(
        """
        **1. Profil utilisateur**

        Le système récupère les ressources déjà utilisées par
        l'utilisateur sélectionné.

        **2. Recherche des utilisateurs similaires**

        Le fichier `user_neighbors.parquet` fournit les
        utilisateurs ayant des profils proches.

        **3. Recherche des nouvelles ressources**

        Le système cherche les ressources utilisées par ces
        utilisateurs similaires mais absentes du profil de
        l'utilisateur cible.

        **4. Score collaboratif**

        Une ressource reçoit davantage de poids lorsqu'elle est
        proposée par des utilisateurs présentant une forte
        similarité.

        **5. Popularité**

        La popularité de la ressource est également prise en compte.

        **6. Score final**

        Le score actuel combine :

        `70 % similarité collaborative + 30 % popularité`

        **7. Filtrage**

        Les ressources déjà utilisées par l'utilisateur sont
        systématiquement retirées des recommandations.
        """
    )

    # --------------------------------------------------------
    # METRIQUES TECHNIQUES
    # --------------------------------------------------------

    st.subheader(
        "📊 Informations techniques"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Utilisateurs",
            interactions["user"].nunique()
        )

    with col2:

        st.metric(
            "Ressources",
            interactions["resource"].nunique()
        )

    with col3:

        st.metric(
            "Interactions",
            len(interactions)
        )

    with col4:

        st.metric(
            "Paires utilisateurs",
            len(neighbors)
        )