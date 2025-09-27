import os
import random
from datetime import datetime, timedelta

from fastapi import FastAPI, APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext

from configuration import collection, news_collection
from models import User, LoginUser
from gmail_service import send_email

app = FastAPI()
user_router = APIRouter(prefix="/users", tags=["Users"])
news_router = APIRouter(prefix="/news", tags=["News"])

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY")
CRON_SECRET = os.getenv("CRON_SECRET")

# ---------------- Request Models ----------------
class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    email: EmailStr
    otp: str
    new_password: str

# ---------------- Email Helpers ----------------
async def send_welcome_email(email: str, full_name: str):
    send_email(email, "Welcome! 🎉", f"<h2>Hello {full_name},</h2><p>Welcome to Fake News Detector.</p>")

async def send_otp_email(email: str, otp: str):
    send_email(email, "Password Reset OTP", f"<p>Your OTP is: <b>{otp}</b></p><p>Valid for 5 minutes.</p>")

# ---------------- User Routes ----------------
@user_router.post("/register")
async def register_user(new_user: User, background_tasks: BackgroundTasks):
    email = new_user.email.strip().lower()
    if collection.find_one({"username": new_user.username}):
        raise HTTPException(status_code=400, detail="Username already taken")
    if collection.find_one({"email": email}):
        raise HTTPException(status_code=400, detail="Email already registered")

    user_dict = new_user.model_dump()
    user_dict["email"] = email
    user_dict["password"] = pwd_context.hash(new_user.password)
    user_dict["created_at"] = datetime.utcnow()
    resp = collection.insert_one(user_dict)

    background_tasks.add_task(send_welcome_email, email, new_user.full_name)

    return {"status": "success", "id": str(resp.inserted_id), "message": "User registered successfully"}

@user_router.post("/forgot-password")
async def forgot_password(request: ForgotPasswordRequest, background_tasks: BackgroundTasks):
    email = request.email.strip().lower()
    user = collection.find_one({"email": email})
    if not user:
        raise HTTPException(status_code=404, detail="Email not registered")

    otp = str(random.randint(100000, 999999))
    expiry = datetime.utcnow() + timedelta(minutes=5)
    collection.update_one({"email": email}, {"$set": {"reset_otp": otp, "reset_expiry": expiry}})

    background_tasks.add_task(send_otp_email, email, otp)
    return {"status": "success", "message": "OTP sent to email"}

# ---------------- Include Routers ----------------
app.include_router(user_router)
app.include_router(news_router)
