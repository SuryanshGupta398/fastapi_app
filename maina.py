from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.responses import JSONResponse
from configuration import collection
from models import User, LoginUser
from datetime import datetime
from passlib.context import CryptContext
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
import os

app = FastAPI()
user_router = APIRouter(prefix="/users", tags=["Users"])

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ------------------ Email Config ------------------
conf = ConnectionConfig(
    MAIL_USERNAME=os.getenv("MAIL_USERNAME", "your_email@gmail.com"),
    MAIL_PASSWORD=os.getenv("MAIL_PASSWORD", "your_app_password"),  # use App Password if Gmail
    MAIL_FROM=os.getenv("MAIL_FROM", "your_email@gmail.com"),
    MAIL_PORT=587,
    MAIL_SERVER="smtp.gmail.com",
    MAIL_TLS=True,
    MAIL_SSL=False,
    USE_CREDENTIALS=True
)

# ------------------ SIGN IN ------------------
@user_router.post("/signin")
async def signin_user(login_user: LoginUser):
    user_in_db = collection.find_one({"email": login_user.email})
    if not user_in_db:
        return JSONResponse(
            status_code=401,
            content={"status_code": 401, "message": "Invalid email or password"}
        )

    if not pwd_context.verify(login_user.password, user_in_db["password"]):
        return JSONResponse(
            status_code=401,
            content={"status_code": 401, "message": "Invalid email or password"}
        )

    return JSONResponse(
        status_code=200,
        content={
            "status_code": 200,
            "message": "Login successful",
            "full_name": user_in_db.get("full_name", ""),
            "username": user_in_db.get("username", ""),
            "email": user_in_db["email"]
        }
    )

# ------------------ REGISTER ------------------
@user_router.post("/register")
async def register_user(new_user: User):
    try:
        if collection.find_one({"username": new_user.username}):
            return JSONResponse(
                status_code=400,
                content={"status_code": 400, "message": "Username already taken"}
            )
        if collection.find_one({"email": new_user.email}):
            return JSONResponse(
                status_code=400,
                content={"status_code": 400, "message": "Email already registered"}
            )

        user_dict = new_user.model_dump()
        user_dict["created_at"] = datetime.utcnow()
        user_dict["password"] = pwd_context.hash(new_user.password)

        resp = collection.insert_one(user_dict)

        # --------- Send Welcome Email ---------
        message = MessageSchema(
            subject="🎉 Welcome to FNDS2!",
            recipients=[new_user.email],
            body=f"Hello {new_user.full_name},\n\nYour account has been created successfully.\n\nThanks for joining us 🚀",
            subtype="plain"
        )
        fm = FastMail(conf)
        await fm.send_message(message)

        return JSONResponse(
            status_code=200,
            content={
                "status_code": 200,
                "id": str(resp.inserted_id),
                "message": "User registered successfully & welcome email sent ✅"
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status_code": 500, "message": f"Some error occurred: {str(e)}"}
        )

# ------------------ DELETE ------------------
@user_router.delete("/delete/{email}")
async def delete_user(email: str):
    result = collection.delete_one({"email": email})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
    return {"status": "success", "message": "Account deleted"}

app.include_router(user_router)
