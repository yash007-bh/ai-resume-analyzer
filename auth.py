import bcrypt
from database import users_collection

def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())

def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed)

def register_user(username, password):
    if users_collection.find_one({"username": username}):
        return False
    hashed = hash_password(password)
    users_collection.insert_one({
        "username": username,
        "password": hashed
    })
    return True

def login_user(username, password):
    user = users_collection.find_one({"username": username})
    if user and verify_password(password, user["password"]):
        return True
    return False