import os
import random
import requests
from datetime import datetime, timedelta

from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig, MessageType
from apscheduler.schedulers.background import BackgroundScheduler

from configuration import collection, news_collection  # MongoDB collections
from models import User, LoginUser

# ------------------ App & Router ------------------
app = FastAPI()
user_router = APIRouter(prefix="/users", tags=["Users"])
news_router = APIRouter(prefix="/news", tags=["News"])

# ------------------ Password Hashing ------------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ------------------ GNews API Key ------------------
GNEWS_API_KEY = os.getenv("GNEWS_KEY")

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

# ------------------ Request Models ------------------
class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    email: EmailStr
    otp: str
    new_password: str

# ------------------ User Routes ------------------
@user_router.post("/register")
async def register_user(new_user: User):
    email = new_user.email.strip().lower()
    if collection.find_one({"username": new_user.username}):
        raise HTTPException(status_code=400, detail="Username already taken")
    if collection.find_one({"email": email}):
        raise HTTPException(status_code=400, detail="Email already registered")

    user_dict = new_user.model_dump()
    user_dict["email"] = email
    user_dict["password"] = pwd_context.hash(new_user.password)
    user_dict["created_at"] = datetime.utcnow()
    collection.insert_one(user_dict)

    # Send welcome email
    message = MessageSchema(
        subject="Welcome to Fake News Detector 🎉",
        recipients=[new_user.email],
        body=f"""<h2>Hello {new_user.full_name},</h2>
        <p>Thank you for signing up with <b>Fake News Detector</b>! We’re excited to have you join our mission of making the internet safer.</p>
        <p>With our app you can:</p>
        <ul>
            <li>✅ Instantly check if a news article is genuine</li>
            <li>✅ Stay updated with verified news sources</li>
            <li>✅ Report suspicious content</li>
        </ul>
        <p>Welcome once again, and thank you for trusting us.</p>
        <br><p>Best regards,<br>The Multiverse Team</p>""",
        subtype=MessageType.html
    )
    await fm.send_message(message)

    return JSONResponse(status_code=200, content={"status": "success", "message": "User registered successfully"})

@user_router.post("/signin")
async def signin_user(login_user: LoginUser):
    email = login_user.email.strip().lower()
    user_in_db = collection.find_one({"email": email})
    if not user_in_db or not pwd_context.verify(login_user.password, user_in_db["password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    user_data = {
        "full_name": user_in_db.get("full_name", ""),
        "username": user_in_db.get("username", ""),
        "email": user_in_db["email"]
    }
    return JSONResponse(status_code=200, content={"status": "success", "message": "Login successful", "user": user_data})

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

@user_router.post("/reset-password")
async def reset_password(request: ResetPasswordRequest):
    email = request.email.strip().lower()
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
    if request.otp != otp_in_db:
        raise HTTPException(status_code=400, detail="Invalid OTP")

    hashed_password = pwd_context.hash(request.new_password)
    collection.update_one(
        {"email": email},
        {"$set": {"password": hashed_password}, "$unset": {"reset_otp": 1, "reset_expiry": 1}}
    )
    return JSONResponse(status_code=200, content={"status": "success", "message": "Password reset successfully"})

@user_router.delete("/delete/{email}")
async def delete_user(email: str):
    email = email.strip().lower()
    result = collection.delete_one({"email": email})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
    return JSONResponse(status_code=200, content={"status": "success", "message": "Account deleted"})

# ------------------ News Functions ------------------
def fetch_and_store_news(lang="en", max_results=20):
    url = f"https://gnews.io/api/v4/top-headlines?country=in&lang={lang}&max={max_results}&apikey={GNEWS_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        articles = response.json().get("articles", [])
        for a in articles:
            news_collection.update_one(
                {"url": a["url"]},  # Check by URL
                {"$set": {
                    "title": a["title"],
                    "description": a["description"],
                    "image": a.get("image", ""),
                    "publishedAt": a["publishedAt"],
                    "language": lang,
                    "source": "GNews",
                    "createdAt": datetime.utcnow()  # always update timestamp to appear on top
                }},
                upsert=True  # Insert if it doesn't exist
            )

def cleanup_old_news():
    one_week_ago = datetime.utcnow() - timedelta(days=7)
    result = news_collection.delete_many({"createdAt": {"$lt": one_week_ago}})
    print(f"Deleted {result.deleted_count} old news articles")
    return result.deleted_count

# ------------------ News Routes ------------------
@news_router.get("/")
def get_news(language: str = "en", limit: int = 20):
    news = list(news_collection.find({"language": language}).sort("createdAt", -1).limit(limit))
    for n in news:
        n["_id"] = str(n["_id"])
    return {"articles": news}

@news_router.post("/refresh")
def refresh_news():
    fetch_and_store_news("en")
    fetch_and_store_news("hi")
    return {"status": "success", "message": "News refreshed"}

# ------------------ Scheduler ------------------
scheduler = BackgroundScheduler()

# Automatic news refresh every 30 minutes (English + Hindi)
scheduler.add_job(lambda: fetch_and_store_news("en"), 'interval', minutes=30)
scheduler.add_job(lambda: fetch_and_store_news("hi"), 'interval', minutes=30)

# Automatic cleanup every 24 hours
scheduler.add_job(cleanup_old_news, 'interval', hours=24)

scheduler.start()

# ------------------ Include Routers ------------------
app.include_router(user_router)
app.include_router(news_router)
