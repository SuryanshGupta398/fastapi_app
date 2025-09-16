from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.responses import JSONResponse
from configuration import collection
from models import User, LoginUser
from datetime import datetime
from bson.objectid import ObjectId
from passlib.context import CryptContext

app = FastAPI()
user_router = APIRouter(prefix="/users", tags=["Users"])

# Password hashing context (passlib)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ------------------ SIGN IN ------------------
@user_router.post("/signin")
async def signin_user(login_user: LoginUser):
    user_in_db = collection.find_one({"email": login_user.email})
    if not user_in_db:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    # Verify hashed password
    if not pwd_context.verify(login_user.password, user_in_db["password"]):
        return JSONResponse(
            status_code=401,
            content={"status_code": 401, "message": "Invalid email or password"}
        )

    return {
        "status_code": 200,
        "message": "Login successful",
        "full_name": user_in_db.get("full_name", ""),
        "username": user_in_db.get("username", ""),
        "email": user_in_db["email"]
    }

# ------------------ REGISTER ------------------
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

        # Hash password before saving
        user_dict["password"] = pwd_context.hash(new_user.password)

        resp = collection.insert_one(user_dict)
        return {
            "status_code": 200,
            "id": str(resp.inserted_id),
            "message": "User registered successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Some error occurred: {e}")

# Include router
app.include_router(user_router)
