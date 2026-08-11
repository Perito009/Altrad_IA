import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# =============================================================================
# CONFIGURATION DE LA PAGE
# =============================================================================

st.set_page_config(
    page_title="Dashboard SharePoint",
    page_icon="📊",
    layout="wide"
)



st.title("📊 Dashboard SharePoint")

st.markdown("""
Bienvenue sur le tableau de bord d'analyse SharePoint.

Cette application permet de :

- Explorer les données
- Filtrer les informations
- Visualiser les statistiques
- Identifier les sites les plus utilisés
- Identifier les utilisateurs les plus présents
""")



@st.cache_data
def load_data():

    """
    Charge le fichier CSV.

    Le cache évite de recharger le fichier
    à chaque interaction.

    Returns
    -------
    pandas.DataFrame
    """

    df = pd.read_csv("Data/BD_Sharepoint_Clean.csv")

    # Suppression des espaces dans les noms de colonnes
    df.columns = df.columns.str.strip()

    return df


# Chargement des données
df = load_data()

# =============================================================================
# APERCU
# =============================================================================

st.subheader("Aperçu des données")

st.dataframe(df.head())

# =============================================================================
# INFORMATIONS SUR LE JEU DE DONNÉES
# =============================================================================

with st.expander("Informations"):

    st.write("Nombre de lignes :", df.shape[0])
    st.write("Nombre de colonnes :", df.shape[1])

    st.write("Colonnes :")

    st.write(df.columns.tolist())

# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.title("Filtres")

# =============================================================================
# CREATION DES FILTRES
# =============================================================================

site = st.sidebar.multiselect(
    "Site",
    sorted(df["site"].dropna().unique())
)

sous_site = st.sidebar.multiselect(
    "Sous-site",
    sorted(df["sous-site"].dropna().unique())
)

bibliotheque = st.sidebar.multiselect(
    "Bibliothèque",
    sorted(df["bibliothèque"].dropna().unique())
)

liste = st.sidebar.multiselect(
    "Liste",
    sorted(df["liste"].dropna().unique())
)

utilisateur = st.sidebar.multiselect(
    "Utilisateur",
    sorted(df["users"].dropna().unique())
)

# =============================================================================
# APPLICATION DES FILTRES
# =============================================================================

df_filtre = df.copy()

if site:
    df_filtre = df_filtre[df_filtre["site"].isin(site)]

if sous_site:
    df_filtre = df_filtre[df_filtre["sous-site"].isin(sous_site)]

if bibliotheque:
    df_filtre = df_filtre[df_filtre["bibliothèque"].isin(bibliotheque)]

if liste:
    df_filtre = df_filtre[df_filtre["liste"].isin(liste)]

if utilisateur:
    df_filtre = df_filtre[df_filtre["users"].isin(utilisateur)]

# =============================================================================
# KPI
# =============================================================================

st.markdown("---")
st.subheader("Indicateurs clés")

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.metric(
        "Sites",
        df_filtre["site"].nunique()
    )

with col2:
    st.metric(
        "Sous-sites",
        df_filtre["sous-site"].nunique()
    )

with col3:
    st.metric(
        "Bibliothèques",
        df_filtre["bibliothèque"].nunique()
    )

with col4:
    st.metric(
        "Listes",
        df_filtre["liste"].nunique()
    )

with col5:
    st.metric(
        "Utilisateurs",
        df_filtre["users"].nunique()
    )

with col6:
    st.metric(
        "Enregistrements",
        len(df_filtre)
    )

st.markdown("---")
st.subheader("Données filtrées")

st.dataframe(df_filtre)