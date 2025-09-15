from fastapi import FastAPI, APIRouter, HTTPException
from configuration import collection
from models import User
from datetime import datetime
from bson.objectid import ObjectId

app = FastAPI()
user_router = APIRouter(prefix="/users", tags=["Users"])

@user_router.post("/register")
async def register_user(new_user: User):
    try:
        # check if username or email already exists
        if collection.find_one({"username": new_user.username}):
            raise HTTPException(status_code=400, detail="Username already taken")
        if collection.find_one({"email": new_user.email}):
            raise HTTPException(status_code=400, detail="Email already registered")

        user_dict = new_user.model_dump()
        user_dict["created_at"] = datetime.utcnow()

        resp = collection.insert_one(user_dict)
        return {
            "status_code": 200,
            "id": str(resp.inserted_id),
            "message": "User registered successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Some error occurred: {e}")


app.include_router(user_router)
