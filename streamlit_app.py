import streamlit as st
from pymongo import MongoClient
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Configuration de la connexion à MongoDB
uri = "votre_chaine_de_connexion_mongodb_atlas"
client = MongoClient(uri)
db = client.maBaseDeDonnees
collection = db.maCollection

def insert_data(nom, age, genre):
    result = collection.insert_one({"nom": nom, "age": age, "genre": genre})
    return result.inserted_id

def get_data():
    return list(collection.find({}))

def update_data(query, new_values):
    result = collection.update_one(query, {"$set": new_values})
    return result.modified_count

def delete_data(query):
    result = collection.delete_one(query)
    return result.deleted_count

def main():
    st.title("Application Altrad IA")

    # Barre latérale pour la navigation
    page = st.sidebar.radio("Aller à", ["Accueil", "Insérer des données", "Visualisation", "Prédictions", "Mettre à jour", "Supprimer"])

    if page == "Accueil":
        st.write("Bienvenue dans l'application Altrad IA!")
    elif page == "Insérer des données":
        st.header("Insérer des données")
        nom = st.text_input("Nom")
        age = st.number_input("Âge", min_value=0, max_value=120)
        genre = st.selectbox("Genre", ["Homme", "Femme"])
        if st.button("Insérer"):
            inserted_id = insert_data(nom, age, genre)
            st.success(f"Donnée insérée avec l'ID: {inserted_id}")
    elif page == "Visualisation":
        st.header("Visualisation des données")

        # Exemple de données pour la visualisation
        data = get_data()
        if data:
            df = pd.DataFrame(data)
            # Conversion du genre en numérique pour la visualisation (optionnel)
            df['genre_numeric'] = df['genre'].map({'Homme': 0, 'Femme': 1})

            # Box plot pour la distribution par genre
            fig, ax = plt.subplots()
            sns.countplot(x='genre', data=df, ax=ax)
            ax.set_title("Current Gender Distribution")
            ax.set_xlabel("Gender")
            ax.set_ylabel("Count")
            st.pyplot(fig)

            # Deuxième visualisation : Nombre d'employés par genre
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
        new_age = st.number_input("Nouvel Âge", min_value=0, max_value=120)
        new_genre = st.selectbox("Nouveau Genre", ["Homme", "Femme"])
        if st.button("Mettre à jour"):
            modified_count = update_data({"nom": old_nom}, {"nom": new_nom, "age": new_age, "genre": new_genre})
            st.success(f"Données mises à jour: {modified_count}")
    elif page == "Supprimer":
        st.header("Supprimer les données")
        nom = st.text_input("Nom à supprimer")
        if st.button("Supprimer"):
            deleted_count = delete_data({"nom": nom})
            st.success(f"Données supprimées: {deleted_count}")

if __name__ == "__main__":
    main()
