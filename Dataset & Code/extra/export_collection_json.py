from bson.json_util import dumps
from pymongo import MongoClient
import json
import pymongo

# MONGO_URL = "mongodb+srv://###########:###########@cluster0.dalhe9w.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
MONGO_URL = "mongodb+srv://101rakibulhasan:01633771417@cluster0.dalhe9w.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

client = pymongo.MongoClient(MONGO_URL)
db = client.get_database('valiant')

# llama3.1-dataset
# yi-dataset
# glm4-dataset
# internlm2-dataset
WORKING_MODEL_COLLECTION = input("Enter the collection name: ")
collection = db.get_collection(WORKING_MODEL_COLLECTION)

if __name__ == '__main__':
    cursor = collection.find({})
    with open(f'{WORKING_MODEL_COLLECTION}.json', 'w') as file:
        json.dump(json.loads(dumps(cursor)), file, indent=4)

    print(f"Exported {WORKING_MODEL_COLLECTION}.json")