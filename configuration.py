import os
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

MONGO_URI = os.getenv("mongodb+srv://user_01:bala@cluster0.qgcatm6.mongodb.net/fake_new_app?retryWrites=true&w=majority&appName=Cluster0")

# Create a new client and connect to the server
client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
db = client["fake_new_app"]
collection=db["users"]
# Send a ping to confirm a successful connection
