from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import judge_mongodb
uri = "mongodb+srv://############:#########@cluster0.dalhe9w.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0&tls=true&tlsAllowInvalidCertificates=true"
# Create a new client and connect to the server
client = MongoClient(uri)
db = client.get_database('valiant')

WORKING_MODEL_COLLECTION = 'llama3.1-dataset'
collection = db.get_collection(WORKING_MODEL_COLLECTION)

try:
    pipeline = [
        {'$match': {'operationType': {'$in': ['insert', 'update']}}}
    ]

    collection.delete_many({})
    judge_mongodb.set_track_data("current_llama_conv_id", 0)
    
    with collection.watch(pipeline) as change_stream:
        for change in change_stream:
            operation_type = change["operationType"]
            if operation_type == 'insert':
                print('insert')
            elif operation_type == 'update':
                print('update')

            
except Exception as e:
    print(e)