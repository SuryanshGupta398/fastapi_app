from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr
from configuration import collection  # your MongoDB collection
from models import User, LoginUser
from datetime import datetime, timedelta
from passlib.context import CryptContext
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig, MessageType
import os, random

app = FastAPI()
user_router = APIRouter(prefix="/users", tags=["Users"])

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ------------------ Email Configuration ------------------
conf = ConnectionConfig(
    MAIL_USERNAME=os.getenv("MAIL_USERNAME"),
    MAIL_PASSWORD=os.getenv("MAIL_PASSWORD"),
    MAIL_FROM=os.getenv("MAIL_FROM"),
    MAIL_PORT=587,
    MAIL_SERVER="smtp.gmail.com",
    MAIL_STARTTLS=True,
    MAIL_SSL_TLS=False,
    USE_CREDENTIALS=True
)

fm = FastMail(conf)

# ------------------ REQUEST MODELS ------------------
class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    email: EmailStr
    otp: str
    new_password: str

# ------------------ REGISTER ------------------
@user_router.post("/register")
async def register_user(new_user: User):
    new_user.email = new_user.email.strip().lower()

    if collection.find_one({"username": new_user.username}):
        raise HTTPException(status_code=400, detail="Username already taken")
    if collection.find_one({"email": new_user.email}):
        raise HTTPException(status_code=400, detail="Email already registered")

    user_dict = new_user.model_dump()
    user_dict["created_at"] = datetime.utcnow()
    user_dict["password"] = pwd_context.hash(new_user.password)

    collection.insert_one(user_dict)

    # Send welcome email
    message = MessageSchema(
        subject="Welcome to Fake News Detector 🎉",
        recipients=[new_user.email],
        body=f"<h2>Hello {new_user.full_name},</h2><p>Thanks for registering!</p>",
        subtype=MessageType.html
    )
    await fm.send_message(message)

    return JSONResponse(status_code=200, content={"status": "success", "message": "User registered successfully"})

# ------------------ SIGN IN ------------------
@user_router.post("/signin")
async def signin_user(login_user: LoginUser):
    email = login_user.email.strip().lower()
    password = login_user.password

    user_in_db = collection.find_one({"email": email})
    if not user_in_db:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not pwd_context.verify(password, user_in_db["password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

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

# ------------------ FORGOT PASSWORD ------------------
@user_router.post("/forgot-password")
async def forgot_password(request: ForgotPasswordRequest):
    email = request.email.strip().lower()
    user = collection.find_one({"email": email})
    if not user:
        raise HTTPException(status_code=404, detail="Email not registered")

    now = datetime.utcnow()
    otp_in_db = user.get("reset_otp")
    expiry_in_db = user.get("reset_expiry")

    if otp_in_db and expiry_in_db and expiry_in_db > now:
        otp_to_send = otp_in_db
    else:
        otp_to_send = str(random.randint(100000, 999999))
        expiry_in_db = now + timedelta(minutes=5)
        collection.update_one(
            {"email": email},
            {"$set": {"reset_otp": otp_to_send, "reset_expiry": expiry_in_db}}
        )

    message = MessageSchema(
        subject="Password Reset OTP",
        recipients=[email],
        body=f"<h2>Password Reset Request</h2><p>Your OTP is: <b>{otp_to_send}</b></p><p>Valid for 5 minutes.</p>",
        subtype=MessageType.html
    )
    await fm.send_message(message)

    return JSONResponse(status_code=200, content={"status": "success", "message": "OTP sent to email"})

# ------------------ RESET PASSWORD ------------------
@user_router.post("/reset-password")
async def reset_password(request: ResetPasswordRequest):
    email = request.email.strip().lower()
    otp = request.otp
    new_password = request.new_password

    user = collection.find_one({"email": email})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    otp_in_db = user.get("reset_otp")
    expiry_in_db = user.get("reset_expiry")

    if not otp_in_db or not expiry_in_db:
        raise HTTPException(status_code=400, detail="OTP not generated. Request a new one.")

    if isinstance(expiry_in_db, str):
        expiry_in_db = datetime.fromisoformat(expiry_in_db)

    now = datetime.utcnow()
    if now > expiry_in_db:
        raise HTTPException(status_code=400, detail="OTP expired. Request a new one.")

    if otp != otp_in_db:
        raise HTTPException(status_code=400, detail="Invalid OTP")

    hashed_password = pwd_context.hash(new_password)
    collection.update_one(
        {"email": email},
        {"$set": {"password": hashed_password}, "$unset": {"reset_otp": 1, "reset_expiry": 1}}
    )

    return JSONResponse(status_code=200, content={"status": "success", "message": "Password reset successfully"})

# ------------------ DELETE ------------------
@user_router.delete("/delete/{email}")
async def delete_user(email: str):
    email = email.strip().lower()
    result = collection.delete_one({"email": email})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="User not found")

    return JSONResponse(status_code=200, content={"status": "success", "message": "Account deleted"})

# ------------------ Include Router ------------------
app.include_router(user_router)
