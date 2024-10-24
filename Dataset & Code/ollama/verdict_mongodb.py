import requests
import pymongo
from bson import ObjectId
import random
import time
import threading

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

print("Initialization MongoDB Done...")

# gets track data from Track Collection
def get_track_data(key, session=None):
    track_data = track.find_one({'_id': ObjectId(track_data__id)}, session=session)
    return track_data[key]

def sysPrint(message):
    print(f"[SYS] {message}")

def errPrint(message):
    print(f"[ERR] {message}")

def get_conv_data(id):
    return collection.find_one({
        "_id" : id
    })

print(get_track_data("current_llama_conv_mongoid"))

def set_verdict_message_db(id, message, response_time):
    message_data = {
        "role": "verdict",
        "content": message,
        "response_time": response_time
    }
    print("THE WAIT IS HERE")
    collection.update_one({"_id": id}, {"$push": {"messages": message_data}})
    print(f"Verdict message added to DB")

def get_current_doc_value(id, key):
    document = get_conv_data(id)
    return document[key]

def gen_verdict_message():
    conv_start_time = time.time()
    judge_message = "I am the verdict!"
    conv_end_time = time.time()
    response_time = conv_end_time - conv_start_time
    return judge_message, response_time