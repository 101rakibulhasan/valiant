from bson.json_util import dumps
from pymongo import MongoClient
import json
from judge_mongodb import collection

if __name__ == '__main__':
    cursor = collection.find({})
    with open('collection.json', 'w') as file:
        json.dump(json.loads(dumps(cursor)), file, indent=4)