from BD_Mongo_co.mongo_connect import inventaire_collection

def insert_inventaire(data):
    result = inventaire_collection.insert_one(data)
    return result.inserted_id

def get_inventaire():
    return list(inventaire_collection.find({}))

def update_inventaire(query, new_values):
    result = inventaire_collection.update_one(query, {"$set": new_values})
    return result.modified_count

def delete_inventaire(query):
    result = inventaire_collection.delete_one(query)
    return result.deleted_count
