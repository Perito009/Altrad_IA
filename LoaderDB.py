import pandas as pd
import sqlite3

def create_database_and_insert_data(csv_file_path, db_name='data.db'):
    # Création d'une connexion à la base de données SQLite
    conn = sqlite3.connect(db_name)
    
    # Lire le fichier CSV en utilisant Pandas
    df = pd.read_csv(csv_file_path)

    # Supposer que le nom de la table est identique au nom du fichier sans l'extension
    table_name = csv_file_path.split('/')[-1].split('.')[0]

    # Créer une table SQLite avec les noms des colonnes du CSV comme noms de colonnes
    df.to_sql(table_name, conn, if_exists='replace', index=False)

    # Fermer la connexion à la base de données
    conn.close()

if __name__ == "__main__":
    csv_file_path = 'Data/BD_Sharepoint_Clean.csv'
    create_database_and_insert_data(csv_file_path)