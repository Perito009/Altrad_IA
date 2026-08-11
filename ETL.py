import pandas as pd

def load_data(file_path):
    """
    Charge les données depuis un fichier CSV.
    
    :param file_path: Le chemin vers le fichier CSV.
    :return: Un DataFrame contenant les données chargées.
    """
    return pd.read_csv(file_path)

# Nettoyer les noms des colonnes
def clean_column_names(df):
    """
    Standardise les noms de colonnes en supprimant les espaces multiples,
    convertir en minuscules et remplacer certains caractères spéciaux.
    
    :param df: DataFrame à nettoyer.
    :return: DataFrame avec les noms des colonnes standardisés.
    """
    return df.rename(columns=lambda x: x.strip().lower().replace(' ', '_'))

# Vérifier et corriger les doublons
def remove_duplicates(df):
    """
    Supprime les lignes en double dans le DataFrame.
    
    :param df: DataFrame à nettoyer.
    :return: DataFrame sans doublons.
    """
    return df.drop_duplicates()

def clean_users_column(df):
    """
    Nettoie la colonne 'Users' en conservant uniquement le prénom, sauf pour les cas spéciaux "Pas de droit" ou des adresses e-mail.
    
    :param df: DataFrame contenant les données avec la colonne 'Users'.
    :return: DataFrame avec la colonne 'users' nettoyée.
    """

    # Ne pas changer si l'utilisateur est exactement "Pas de droit"
    def dont_change_it(user):
        if user == 'pas de droit':
            return 'Pas de droit'
        
        return None
    
    # Extraire le prénom
    def get_first_name(full_name):
        parts = full_name.split(' ')
        return parts[0].strip() if len(parts) > 0 else ''

    # Extraire le nom avant "@"
    def get_username_before_at(email):
        return email.split('@')[0] if '@' in email else 'Inconnu'
    
    # Affichage des valeurs manquantes
    missing_values_count = df['Users'].isna().sum()
    total_entries = len(df)
    missing_percentage = (missing_values_count / total_entries) * 100
    print(f"Pourcentage de valeurs manquantes : {missing_percentage:.2f}%")
    
    # Affichage des valeurs commençant par 'i:0#.f'
    values_starting_with_pattern = df[df['Users'].str.startswith('i:0#.f')]
    pattern_count = len(values_starting_with_pattern)
    total_users = len(df)
    pattern_percentage = (pattern_count / total_users) * 100
    print(f"Pourcentage de valeurs commençant par i:0#.f : {pattern_percentage:.2f}%")

    # Remplacer les valeurs qui commencent par 'i:0#.f|membership|' par "Inconnu"
    df['Users'] = df['Users'].apply(lambda x: 'Inconnu' if str(x).startswith('i:0#.f') else x)
    
    # Appliquer la fonction pour gérer les adresses e-mail et noms spéciaux
    def clean_user_name(user):
        if dont_change_it(user) is not None:
            return dont_change_it(user)
        
        if '@' in user:
            return get_username_before_at(user)
        else:
            return get_first_name(user)

    df['Users'] = df['Users'].apply(clean_user_name)
    
    return df


# Enregistrer les données nettoyées
def save_cleaned_data(df, file_path):
    """
    Sauvegarde le DataFrame nettoyé dans un nouveau fichier CSV.
    
    :param df: DataFrame à sauvegarder.
    :param file_path: Chemin vers le fichier CSV de sortie.
    """
    # Enregistrer les données nettoyées dans un nouveau fichier
    df.to_csv(file_path, index=False)

# Fonction principale pour lancer la procédure de nettoyage
def main():
    # Charger les données du fichier CSV
    file_path = 'Data/Refactor_BD_Sharepoint.csv'
    df = load_data(file_path)
    
    # Afficher un aperçu des données chargées
    print(df.head())
    
    # Standardiser le nom de la colonne Users
    df['Users'] = df['Users'].str.lower().str.strip()
    
    # Nettoyer la colonne 'users'
    df_cleaned_users = clean_users_column(df)

    # Nettoyer les noms des colonnes
    df_cleaned_columns = clean_column_names(df_cleaned_users)
    
    # Supprimer les doublons (sur le DataFrame avec les noms de colonnes nettoyés et la colonne Users propre)
    data_no_duplicates = remove_duplicates(df_cleaned_columns)

    # Sauvegarder les données nettoyées dans un nouveau fichier CSV
    save_cleaned_data(data_no_duplicates, 'Data/BD_Sharepoint_Clean.csv')

if __name__ == "__main__":
    main()