from pymongo import MongoClient

# Connexion à MongoDB
uri = "mongodb+srv://olimbo:<pCAlSp6ExGy6P4Z5>@cluster0.zokzm12.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri)
db = client['db_AltradIA']

# Accès aux collections
inventaire_collection = db['inventaire']
sharepoint_collection = db['bd_sharepoint']
effectif_collection = db['effectif']
