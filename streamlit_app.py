import streamlit as st
import pandas as pd
import plotly.express as px
import importlib.util
import sys

from pathlib import Path

# Import du moteur de recommandation
recommender_path = Path(__file__).resolve().parent / "SharePoint-Recommender" / "recommender.py"
if recommender_path.exists():
    spec = importlib.util.spec_from_file_location("recommender_module", recommender_path)
    recommender_module = importlib.util.module_from_spec(spec)
    sys.modules["recommender_module"] = recommender_module
    spec.loader.exec_module(recommender_module)
    load_user_resource_matrix = recommender_module.load_user_resource_matrix
    load_cosine_similarity = recommender_module.load_cosine_similarity
    get_user_resources = recommender_module.get_user_resources
    get_similar_users = recommender_module.get_similar_users
    recommend_resources = recommender_module.recommend_resources
    popular_resources = recommender_module.popular_resources
    explain_recommendation = recommender_module.explain_recommendation
else:
    raise ImportError(f"Cannot find recommender module at {recommender_path}")


# =============================================================================
# CONFIGURATION DE LA PAGE
# =============================================================================

st.set_page_config(
    page_title="Dashboard SharePoint",
    page_icon="📊",
    layout="wide"
)


# =============================================================================
# STYLE
# =============================================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: bold;
    }

    .subtitle {
        font-size: 18px;
        color: #666666;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =============================================================================
# TITRE
# =============================================================================

st.markdown(
    '<div class="main-title">📊 Dashboard SharePoint</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Analyse et recommandation des ressources SharePoint.
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    Cette application permet de :

    - 🔎 Explorer les données SharePoint
    - 🎛️ Filtrer les informations
    - 📊 Visualiser les statistiques
    - 👥 Identifier les utilisateurs les plus présents
    - 🌐 Identifier les sites les plus utilisés
    - 🤖 Obtenir des recommandations personnalisées
    """
)


st.divider()


# =============================================================================
# CHARGEMENT DU CSV
# =============================================================================

@st.cache_data
def load_data():
    """
    Charge le fichier CSV SharePoint.

    Le cache permet d'éviter de relire le fichier
    à chaque interaction avec l'application.
    """

    file_path = Path(
        "Data/Cleaned_BD_Sharepoint.csv"
    )

    if not file_path.exists():

        raise FileNotFoundError(
            f"Fichier introuvable : {file_path}"
        )

    df = pd.read_csv(file_path)

    # Suppression des espaces dans les noms de colonnes.
    df.columns = df.columns.str.strip()

    return df


# =============================================================================
# CHARGEMENT
# =============================================================================

try:

    df = load_data()

except Exception as error:

    st.error(
        f"Erreur lors du chargement du CSV : {error}"
    )

    st.stop()


# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.title("🎛️ Navigation")


page = st.sidebar.radio(
    "Choisir une section",
    [
        "📊 Dashboard",
        "🤖 Recommandation",
    ]
)


# =============================================================================
# PAGE DASHBOARD
# =============================================================================

if page == "📊 Dashboard":

    # -------------------------------------------------------------------------
    # APERCU
    # -------------------------------------------------------------------------

    st.subheader("👀 Aperçu des données")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


    # -------------------------------------------------------------------------
    # INFORMATIONS
    # -------------------------------------------------------------------------

    with st.expander("ℹ️ Informations sur le dataset"):

        st.write(
            "Nombre de lignes :",
            df.shape[0]
        )

        st.write(
            "Nombre de colonnes :",
            df.shape[1]
        )

        st.write(
            "Colonnes :",
            df.columns.tolist()
        )


    # =========================================================================
    # FILTRES
    # =========================================================================

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔎 Filtres")


    # -------------------------------------------------------------------------
    # FILTRE SITE
    # -------------------------------------------------------------------------

    site = st.sidebar.multiselect(
        "Site",
        sorted(
            df["site"]
            .dropna()
            .unique()
        )
    )


    # -------------------------------------------------------------------------
    # FILTRE SOUS-SITE
    # -------------------------------------------------------------------------

    sous_site = st.sidebar.multiselect(
        "Sous-site",
        sorted(
            df["sous-site"]
            .dropna()
            .unique()
        )
    )


    # -------------------------------------------------------------------------
    # FILTRE BIBLIOTHEQUE
    # -------------------------------------------------------------------------

    bibliotheque = st.sidebar.multiselect(
        "Bibliothèque",
        sorted(
            df["bibliothèque"]
            .dropna()
            .unique()
        )
    )


    # -------------------------------------------------------------------------
    # FILTRE LISTE
    # -------------------------------------------------------------------------

    liste = st.sidebar.multiselect(
        "Liste",
        sorted(
            df["liste"]
            .dropna()
            .unique()
        )
    )


    # -------------------------------------------------------------------------
    # FILTRE UTILISATEUR
    # -------------------------------------------------------------------------

    utilisateur = st.sidebar.multiselect(
        "Utilisateur",
        sorted(
            df["users"]
            .dropna()
            .unique()
        )
    )


    # =========================================================================
    # APPLICATION DES FILTRES
    # =========================================================================

    df_filtre = df.copy()


    if site:

        df_filtre = df_filtre[
            df_filtre["site"].isin(site)
        ]


    if sous_site:

        df_filtre = df_filtre[
            df_filtre["sous-site"].isin(
                sous_site
            )
        ]


    if bibliotheque:

        df_filtre = df_filtre[
            df_filtre["bibliothèque"].isin(
                bibliotheque
            )
        ]


    if liste:

        df_filtre = df_filtre[
            df_filtre["liste"].isin(
                liste
            )
        ]


    if utilisateur:

        df_filtre = df_filtre[
            df_filtre["users"].isin(
                utilisateur
            )
        ]


    # =========================================================================
    # KPI
    # =========================================================================

    st.markdown("---")

    st.subheader(
        "📌 Indicateurs clés"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "🌐 Sites",
            df_filtre["site"].nunique()
        )


    with col2:

        st.metric(
            "👥 Utilisateurs",
            df_filtre["users"].nunique()
        )


    with col3:

        st.metric(
            "📄 Enregistrements",
            len(df_filtre)
        )


    col4, col5, col6 = st.columns(3)


    with col4:

        st.metric(
            "📁 Sous-sites",
            df_filtre["sous-site"].nunique()
        )


    with col5:

        st.metric(
            "📚 Bibliothèques",
            df_filtre["bibliothèque"].nunique()
        )


    with col6:

        st.metric(
            "📋 Listes",
            df_filtre["liste"].nunique()
        )


    st.markdown("---")


    # =========================================================================
    # GRAPHIQUES
    # =========================================================================

    st.subheader(
        "📈 Analyse des données"
    )


    col1, col2 = st.columns(2)


    # -------------------------------------------------------------------------
    # TOP SITES
    # -------------------------------------------------------------------------

    with col1:

        site_counts = (
            df_filtre["site"]
            .value_counts()
            .head(10)
            .reset_index()
        )

        site_counts.columns = [
            "site",
            "nombre"
        ]


        fig_site = px.bar(
            site_counts,
            x="nombre",
            y="site",
            orientation="h",
            title="🌐 Top 10 des sites"
        )


        st.plotly_chart(
            fig_site,
            use_container_width=True
        )


    # -------------------------------------------------------------------------
    # TOP UTILISATEURS
    # -------------------------------------------------------------------------

    with col2:

        user_counts = (
            df_filtre["users"]
            .value_counts()
            .head(10)
            .reset_index()
        )

        user_counts.columns = [
            "users",
            "nombre"
        ]


        fig_users = px.bar(
            user_counts,
            x="nombre",
            y="users",
            orientation="h",
            title="👥 Top 10 des utilisateurs"
        )


        st.plotly_chart(
            fig_users,
            use_container_width=True
        )


    # =========================================================================
    # BIBLIOTHEQUES
    # =========================================================================

    st.subheader(
        "📚 Bibliothèques les plus utilisées"
    )


    library_counts = (
        df_filtre["bibliothèque"]
        .value_counts()
        .head(15)
        .reset_index()
    )


    library_counts.columns = [
        "bibliothèque",
        "nombre"
    ]


    fig_library = px.bar(
        library_counts,
        x="nombre",
        y="bibliothèque",
        orientation="h",
        title="Top 15 des bibliothèques"
    )


    st.plotly_chart(
        fig_library,
        use_container_width=True
    )


    # =========================================================================
    # DONNEES FILTREES
    # =========================================================================

    st.markdown("---")

    st.subheader(
        "📋 Données filtrées"
    )


    st.write(
        f"{len(df_filtre):,} enregistrements"
    )


    st.dataframe(
        df_filtre,
        use_container_width=True
    )


# =============================================================================
# PAGE RECOMMANDATION
# =============================================================================

elif page == "🤖 Recommandation":

    st.header(
        "🤖 Système de recommandation SharePoint"
    )


    st.markdown(
        """
        Le moteur analyse les ressources utilisées par les utilisateurs
        et recherche les utilisateurs ayant des comportements similaires.

        Il recommande ensuite des ressources que l'utilisateur actuel
        n'utilise pas encore.
        """
    )


    st.divider()


    # =========================================================================
    # CHARGEMENT DU MODELE
    # =========================================================================

    try:

        user_resource_matrix = (
            load_user_resource_matrix()
        )

        cosine_matrix = (
            load_cosine_similarity()
        )

    except FileNotFoundError as error:

        st.error(
            str(error)
        )

        st.info(
            """
            Pour initialiser le moteur de recommandation, exécute :

            `python preprocessing.py`

            puis :

            `python similarity.py`
            """
        )

        st.stop()


    # =========================================================================
    # SELECTION UTILISATEUR
    # =========================================================================

    st.subheader(
        "👤 Sélection de l'utilisateur"
    )


    users = sorted(
        user_resource_matrix.index.tolist()
    )


    selected_user = st.selectbox(
        "Utilisateur",
        users
    )


    # =========================================================================
    # PARAMETRES
    # =========================================================================

    col1, col2 = st.columns(2)


    with col1:

        n_similar_users = st.slider(
            "Nombre d'utilisateurs similaires",
            min_value=3,
            max_value=30,
            value=10
        )


    with col2:

        n_recommendations = st.slider(
            "Nombre de recommandations",
            min_value=5,
            max_value=30,
            value=10
        )


    # =========================================================================
    # RESSOURCES ACTUELLES
    # =========================================================================

    current_resources = (
        get_user_resources(
            selected_user,
            user_resource_matrix
        )
    )


    st.subheader(
        "📚 Ressources actuellement utilisées"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Ressources utilisées",
            len(current_resources)
        )


    with col2:

        st.metric(
            "Ressources disponibles",
            len(
                user_resource_matrix.columns
            )
            - len(current_resources)
        )


    # =========================================================================
    # UTILISATEURS SIMILAIRES
    # =========================================================================

    st.subheader(
        "👥 Utilisateurs similaires"
    )


    similar_users = get_similar_users(
        selected_user,
        cosine_matrix,
        n=n_similar_users
    )


    if similar_users.empty:

        st.warning(
            "Aucun utilisateur similaire trouvé."
        )

    else:

        similar_display = (
            similar_users.copy()
        )


        similar_display[
            "similarity"
        ] = (
            similar_display[
                "similarity"
            ] * 100
        ).round(2)


        similar_display.columns = [
            "Utilisateur",
            "Similarité (%)"
        ]


        st.dataframe(
            similar_display,
            use_container_width=True,
            hide_index=True
        )


    # =========================================================================
    # RECOMMANDATIONS
    # =========================================================================

    st.divider()

    st.subheader(
        "🎯 Recommandations personnalisées"
    )


    recommendations = recommend_resources(
        user=selected_user,
        similarity_matrix=cosine_matrix,
        user_resource_matrix=user_resource_matrix,
        n_users=n_similar_users,
        n_recommendations=n_recommendations
    )


    if recommendations.empty:

        st.warning(
            "Aucune recommandation disponible."
        )

    else:

        st.success(
            f"{len(recommendations)} "
            "recommandations générées."
        )


        # ---------------------------------------------------------------------
        # TABLEAU
        # ---------------------------------------------------------------------

        recommendation_display = (
            recommendations.copy()
        )


        recommendation_display[
            "score_percent"
        ] = (
            recommendation_display[
                "score_percent"
            ].round(2)
        )


        recommendation_display[
            "average_similarity"
        ] = (
            recommendation_display[
                "average_similarity"
            ] * 100
        ).round(2)


        recommendation_display.columns = [
            "Ressource",
            "Score",
            "Utilisateurs support",
            "Similarité moyenne (%)",
            "Score normalisé (%)"
        ]


        st.dataframe(
            recommendation_display,
            use_container_width=True,
            hide_index=True
        )


        # ---------------------------------------------------------------------
        # GRAPHIQUE DES SCORES
        # ---------------------------------------------------------------------

        chart_data = (
            recommendations
            .sort_values(
                "score_percent"
            )
        )


        fig = px.bar(
            chart_data,
            x="score_percent",
            y="resource",
            orientation="h",
            title=(
                "🎯 Score des recommandations"
            ),
            labels={
                "score_percent": "Score (%)",
                "resource": "Ressource"
            }
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


        # =========================================================================
        # EXPLICATION
        # =========================================================================

        st.subheader(
            "💡 Explication de la recommandation"
        )


        selected_resource = st.selectbox(
            "Sélectionner une ressource",
            recommendations[
                "resource"
            ].tolist()
        )


        explanation = explain_recommendation(
            resource=selected_resource,
            user=selected_user,
            similarity_matrix=cosine_matrix,
            user_resource_matrix=user_resource_matrix,
            n_users=n_similar_users
        )


        st.info(
            f"La ressource **{selected_resource}** "
            f"est utilisée par "
            f"**{explanation['support_count']}** "
            f"utilisateur(s) similaire(s) à "
            f"**{selected_user}**."
        )


        # ---------------------------------------------------------------------
        # UTILISATEURS SUPPORTANT LA RECOMMANDATION
        # ---------------------------------------------------------------------

        if explanation[
            "supporting_users"
        ]:

            support_df = pd.DataFrame(
                explanation[
                    "supporting_users"
                ]
            )


            support_df[
                "similarity"
            ] = (
                support_df[
                    "similarity"
                ] * 100
            ).round(2)


            support_df.columns = [
                "Utilisateur",
                "Similarité (%)"
            ]


            st.dataframe(
                support_df,
                use_container_width=True,
                hide_index=True
            )


    # =========================================================================
    # COMPARAISON AVEC LA POPULARITE
    # =========================================================================

    st.divider()

    st.subheader(
        "🔥 Comparaison avec les ressources populaires"
    )


    popular = popular_resources(
        selected_user,
        user_resource_matrix,
        n=n_recommendations
    )


    if popular.empty:

        st.info(
            "Aucune ressource populaire disponible."
        )

    else:

        st.dataframe(
            popular,
            use_container_width=True,
            hide_index=True
        )


    st.caption(
        "La popularité constitue une baseline. "
        "Le moteur personnalisé tient compte du comportement "
        "des utilisateurs similaires."
    )