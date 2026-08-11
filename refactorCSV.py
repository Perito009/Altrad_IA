import pandas as pd

def remplacer_vides_par_dernier_site_sous_site(df):
    """
    Remplacer les '0' dans les colonnes "Site", "Sous-Site" et également dans les colonnes 
    "Liste", "Bibliothèque", et "Users" par la dernière valeur non nulle précédente ou le mot "Aucun".
    
    :param df: DataFrame contenant les données du CSV.
    :return: DataFrame avec les valeurs remplacées.
    """
    previous_site = None
    previous_sous_site = None
    
    for index, row in df.iterrows():
        if isinstance(row['Site'], str) and row['Site'] == '0':
            row['Site'] = previous_site
        else:
            previous_site = row['Site']
        
        if isinstance(row['Sous-Site'], str) and row['Sous-Site'] == '0':
            row['Sous-Site'] = previous_sous_site
        else:
            previous_sous_site = row['Sous-Site']

        # Remplacer les valeurs '0' dans "Liste", "Bibliothèque" et "Users" par "Aucun"
        for column in ['Liste', 'Bibliothèque', 'Users']:
            if isinstance(row[column], str) and row[column] == '0':
                row[column] = 'aucun'
    
    return df

# Chargement du CSV
df = pd.read_csv('Data/BD_Sharepoint.csv')

# Appel de la fonction pour remplacer les valeurs
df_modifie = remplacer_vides_par_dernier_site_sous_site(df)

# Afficher le DataFrame modifié (pour vérification)
print(df_modifie)


df_modifie.to_csv('Data/Refactor_BD_Sharepoint.csv', index=False)