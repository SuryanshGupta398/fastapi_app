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
    MAIL_USERNAME=os.getenv("MAIL_USERNAME", "guptajisuryansh286@gmail.com"),
    MAIL_PASSWORD=os.getenv("MAIL_PASSWORD", "gceq mwac lcqy qytd"),  # use App Password if Gmail
    MAIL_FROM=os.getenv("MAIL_FROM", "guptajisuryansh286@gmail.com"),
    MAIL_PORT=587,
    MAIL_SERVER="smtp.gmail.com",
    MAIL_STARTTLS=True,
    MAIL_SSL_TLS=False,
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
        # check existing
        if collection.find_one({"username": new_user.username}):
            return JSONResponse(status_code=400, content={"message": "Username already taken"})
        if collection.find_one({"email": new_user.email}):
            return JSONResponse(status_code=400, content={"message": "Email already registered"})

        # hash password and save
        user_dict = new_user.model_dump()
        user_dict["created_at"] = datetime.utcnow()
        user_dict["password"] = pwd_context.hash(new_user.password)
        resp = collection.insert_one(user_dict)

        # 📧 send welcome email
        message = MessageSchema(
            subject="Welcome to My App 🎉",
            recipients=[new_user.email],   # send to the registered user
            body=f"""
                <h2>Hello {new_user.full_name},</h2>
                <p>Thank you for registering with us!</p>
                <p>Your username is: <b>{new_user.username}</b></p>
                <br>
                <p>We’re glad to have you onboard 🚀</p>
            """,
            subtype=MessageType.html
        )

        fm = FastMail(conf)
        await fm.send_message(message)

        return JSONResponse(
            status_code=200,
            content={"status": "success", "id": str(resp.inserted_id), "message": "User registered successfully and email sent"}
        )

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Error: {str(e)}"})


# ------------------ DELETE ------------------
@user_router.delete("/delete/{email}")
async def delete_user(email: str):
    result = collection.delete_one({"email": email})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
    return {"status": "success", "message": "Account deleted"}

app.include_router(user_router)
