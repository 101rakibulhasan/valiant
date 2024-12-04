from bson.json_util import dumps
import json
import pymongo

MONGO_URL = "mongodb+srv://101rakibulhasan:01633771417@cluster0.dalhe9w.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = pymongo.MongoClient(MONGO_URL)
db = client.get_database('valiant')

WORKING_MODEL_COLLECTION = 'user-dataset-test-3'
collection = db.get_collection(WORKING_MODEL_COLLECTION)

if __name__ == '__main__':
    cursor = collection.find({})
    with open('collection.json', 'w') as file:
        json.dump(json.loads(dumps(cursor)), file, indent=4)