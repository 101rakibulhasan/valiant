import pymongo
from bson import ObjectId
import random
from util import sysPrint, errPrint

import os
from dotenv import load_dotenv
load_dotenv()

MONGODB_USER = os.getenv('MONGODB_USER')
MONGODB_PASS= os.getenv('MONGODB_PASS')

print("Initializing MongoDB...")
MONGO_URL = f"mongodb+srv://{MONGODB_USER}:{MONGODB_PASS}@cluster0.dalhe9w.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

client = pymongo.MongoClient(MONGO_URL)
db = client.get_database('valiant')

WORKING_MODEL_COLLECTION = 'llama3.1-dataset'
collection = db.get_collection(WORKING_MODEL_COLLECTION)

track = db.get_collection('track')
track_data__id = '66e40b5364eb4675cf7c603f'
track_data = track.find_one({'_id': ObjectId(track_data__id)})

sysPrint("Initialization MongoDB Done...")

# gets track data from Track Collection
def get_track_data(key, session=None):
    track_data = track.find_one({'_id': ObjectId(track_data__id)}, session=session)
    return track_data[key]

# sets track data in Track Collection
def set_track_data(key, value, session=None):
    track.update_one({'_id': ObjectId(track_data__id)}, {'$set': {key: value}}, session=session)

# creates conversation dataset in Velient Collection
def create_conv_dataset(c_id):
    start_conversation = random.randint(0, 1)
    if start_conversation == 0:
        conv_start_by = "judge"
    else:
        conv_start_by = "verdict"

    # dev: delete this
    conv_start_by = "verdict"

    data = {
        "id" : c_id,
        "conv_start_by" : conv_start_by,
        "messages" : [],
        "result" : "AI"
    }

    result = collection.insert_one(data)
    set_track_data("current_llama_conv_mongoid", result.inserted_id)
    return result.inserted_id, conv_start_by

def get_conv_data(id):
    return collection.find_one({
        "_id" : id
    })

def set_judge_message_db(id, message, response_time):
    message_data = {
        "role": "judge",
        "content": message,
        "response_time": response_time
    }
    collection.update_one({"_id": id}, {"$push": {"messages": message_data}})


   
       

