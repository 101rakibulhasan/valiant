import pymongo
from bson import ObjectId

run_conversation = True

MONGO_URL = "mongodb+srv://101rakibulhasan:01633771417@cluster0.dalhe9w.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

client = pymongo.MongoClient(MONGO_URL)
db = client.get_database('valiant')

WORKING_MODEL_COLLECTION = 'llama3.1-dataset'
collection = db.get_collection(WORKING_MODEL_COLLECTION)

track = db.get_collection('track')
track_data__id = '66e40b5364eb4675cf7c603f'

# gets track data from Track Collection
def get_track_data(key, session):
    track_data = track.find_one({'_id': ObjectId(track_data__id)}, session=session)
    return track_data[key]

# sets track data in Track Collection
def set_track_data(key, value, session):
    track.update_one({'_id': ObjectId(track_data__id)}, {'$set': {key: value}}, session=session)
    print(f"Track data updated")

session = client.start_session()
with session.start_transaction():
    current_id = get_track_data("current_llama_conv_id", session)
    # start process
    while current_id < get_track_data("max_llama_conv", session):
        current_id = current_id+1
        set_track_data("current_llama_conv_id", current_id, session)
        print(f"fl {get_track_data("current_llama_conv_id", session)}")
