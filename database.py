import os
from pymongo import MongoClient

mongo_uri = os.environ.get("MONGO_URI")

client = MongoClient(mongo_uri)

db = client["resume_ai_db"]

users_collection = db["users"]
results_collection = db["results"]