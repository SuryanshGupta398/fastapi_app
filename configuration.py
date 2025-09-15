import os
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

MONGO_URI = os.getenv("MONGO_URI")

# Create a new client and connect to the server
client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
db = client["fake_new_app"]
collection=db["users"]
# Send a ping to confirm a successful connection
