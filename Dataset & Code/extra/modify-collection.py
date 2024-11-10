from bson.json_util import dumps
import json
import pymongo

# MONGO_URL = "mongodb+srv://###########:###########@cluster0.dalhe9w.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
MONGO_URL = "mongodb+srv://101rakibulhasan:01633771417@cluster0.dalhe9w.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = pymongo.MongoClient(MONGO_URL)
db = client.get_database('valiant')

def reset_collection():
    collection_name = input("Enter the collection name: ")  
    collection = db.get_collection(collection_name)
    collection.delete_many({})
    print("Collection reset")


def rename_collection():
    prev = input("Enter the previous collection name: ")
    new = input("Enter the new collection name: ")
    db[prev].rename(new)
    print(f"{prev} Collection renamed to {new}")

print("Choose an option:")
print("1. Reset Collection")
print("2. Rename Collection")

option = int(input("Enter your choice: "))

if option == 1:
    reset_collection()
elif option == 2:
    rename_collection()
else:
    print("Invalid choice")