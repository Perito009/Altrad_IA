import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Import de la connexion MongoDB depuis mongo_connect.py
from BD_Mongo_co.mongo_connect import inventaire_collection, sharepoint_collection, effectif_collection
from api.inventaire_api import insert_inventaire, get_inventaire, update_inventaire, delete_inventaire

def main():
    st.title("Application Altrad IA")
    page = st.sidebar.radio("Aller à", ["Accueil", "Insérer des données", "Visualisation", "Prédictions", "Mettre à jour", "Supprimer"])

    if page == "Accueil":
        st.write("Bienvenue dans l'application Altrad IA!")
    elif page == "Insérer des données":
        st.header("Insérer des données")
        nom = st.text_input("Nom")
        prenom = st.text_input("Prénom")
        nature_contrat = st.selectbox("Nature du contrat", ["Contrat à durée indéterminée", "Contrat à durée déterminée"])
        emploi = st.selectbox("Emploi", ["PEINTRE", "CHEF D EQUIPE", "MONTEUR ECHAFAUDEUR", "QHSE MANAGER ANGOLA AFRICA REGIO", "RESP POLE ADMINISTRATIF"])
        debut_contrat = st.date_input("Début du contrat")
        cat_remuneration = st.selectbox("Catégorie rémunération", ["ETAM ART 36 FORFAIT", "AUTRE"])
        affectation1 = st.text_input("Affectation 1", value="GRAND PROJET")
        affectation2 = st.text_input("Affectation 2", value="BU GRAND PROJET INDU")
        genre = st.selectbox("Genre", ["Homme", "Femme"])
        if st.button("Insérer"):
            doc = {
                "Nom": nom,
                "Prénom": prenom,
                "L nature contrat": nature_contrat,
                "L Emploi": emploi,
                "D Début contrat": debut_contrat.strftime("%Y-%m-%d"),
                "L Cat. rémunération": cat_remuneration,
                "L Affectation 1": affectation1,
                "L Affectation 2": affectation2,
                "genre": genre
            }
            inserted_id = insert_inventaire(doc)
            st.success(f"Donnée insérée avec l'ID: {inserted_id}")
    elif page == "Visualisation":
        st.header("Visualisation des données")
        data = get_inventaire()
        if data:
            df = pd.DataFrame(data)
            df['genre_numeric'] = df['genre'].map({'Homme': 0, 'Femme': 1})
            df_dict = df.where(pd.notnull(df), None).to_dict(orient='records')
            fig, ax = plt.subplots()
            sns.countplot(x='genre', data=df, ax=ax)
            ax.set_title("Current Gender Distribution")
            ax.set_xlabel("Gender")
            ax.set_ylabel("Count")
            st.pyplot(fig)
            fig2, ax2 = plt.subplots()
            sns.countplot(x='genre', data=df, ax=ax2)
            ax2.set_title('Nombre d\'employés par genre (Inventaire)')
            ax2.set_xlabel("Genre")
            ax2.set_ylabel("Compte")
            st.pyplot(fig2)
        else:
            st.warning("Aucune donnée disponible pour la visualisation.")
    elif page == "Prédictions":
        st.header("Prédictions des besoins futurs")

        # Exemple de données pour les prédictions
        materials = ['Type A', 'Type B', 'Type C']
        counts = np.random.randint(100, size=len(materials))

        fig, ax = plt.subplots()
        ax.bar(materials, counts)
        ax.set_title('Predicted Future Material Needs')
        ax.set_xlabel('Material Type')
        ax.set_ylabel('Estimated Count')
        st.pyplot(fig)

        # Évolution des effectifs
        years = np.arange(2020, 2031)
        employees = np.random.randint(50, size=len(years)) + np.arange(len(years)) * 5

        fig2, ax2 = plt.subplots()
        ax2.plot(years, employees)
        ax2.set_title('Évolution des effectifs (historique et prédictions)')
        ax2.set_xlabel('Année')
        ax2.set_ylabel('Nombre d\'employés')
        st.pyplot(fig2)
    elif page == "Mettre à jour":
        st.header("Mettre à jour les données")
        old_nom = st.text_input("Nom actuel")
        new_nom = st.text_input("Nouveau Nom")
        new_prenom = st.text_input("Nouveau Prénom")
        new_nature_contrat = st.selectbox("Nouvelle nature du contrat", ["Contrat à durée indéterminée", "Contrat à durée déterminée"])
        new_emploi = st.selectbox("Nouvel Emploi", ["PEINTRE", "CHEF D EQUIPE", "MONTEUR ECHAFAUDEUR", "QHSE MANAGER ANGOLA AFRICA REGIO", "RESP POLE ADMINISTRATIF"])
        new_debut_contrat = st.date_input("Nouveau début du contrat")
        new_cat_remuneration = st.selectbox("Nouvelle catégorie rémunération", ["ETAM ART 36 FORFAIT", "AUTRE"])
        new_affectation1 = st.text_input("Nouvelle Affectation 1", value="GRAND PROJET")
        new_affectation2 = st.text_input("Nouvelle Affectation 2", value="BU GRAND PROJET INDU")
        new_genre = st.selectbox("Nouveau Genre", ["Homme", "Femme"])
        if st.button("Mettre à jour"):
            query = {"Nom": old_nom}
            new_values = {
                "Nom": new_nom,
                "Prénom": new_prenom,
                "L nature contrat": new_nature_contrat,
                "L Emploi": new_emploi,
                "D Début contrat": new_debut_contrat.strftime("%Y-%m-%d"),
                "L Cat. rémunération": new_cat_remuneration,
                "L Affectation 1": new_affectation1,
                "L Affectation 2": new_affectation2,
                "genre": new_genre
            }
            modified_count = update_inventaire(query, new_values)
            st.success(f"Données mises à jour: {modified_count}")
    elif page == "Supprimer":
        st.header("Supprimer les données")
        nom = st.text_input("Nom à supprimer")
        if st.button("Supprimer"):
            deleted_count = delete_inventaire({"Nom": nom})
            st.success(f"Données supprimées: {deleted_count}")

if __name__ == "__main__":
    main()

