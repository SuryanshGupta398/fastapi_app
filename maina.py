import os
import random
import requests
from datetime import datetime, timedelta
from fastapi import FastAPI, APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext

from configuration import collection, news_collection  # MongoDB collections
from models import User, LoginUser
from gmail_service import send_email  # Gmail API helper

# ---------------- FastAPI setup ----------------
app = FastAPI()
user_router = APIRouter(prefix="/users", tags=["Users"])
news_router = APIRouter(prefix="/news", tags=["News"])

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ---------------- Environment ----------------
NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY")
CRON_SECRET = os.getenv("CRON_SECRET")

# ---------------- Request Models ----------------
class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    email: EmailStr
    otp: str
    new_password: str

# ---------------- Health Routes ----------------
@app.get("/health")
def health_check():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}

@app.head("/health")
def health_check_head():
    return {"status": "ok"}

# ---------------- Email Helpers ----------------
async def send_welcome_email(email: str, full_name: str):
    subject = "Welcome to Fake News Detector 🎉"
    body = f"<h2>Hello {full_name},</h2><p>Thank you for signing up!</p>"
    # run blocking send_email in background
    from fastapi.concurrency import run_in_threadpool
    await run_in_threadpool(send_email, email, subject, body)

async def send_otp_email(email: str, otp: str):
    subject = "Password Reset OTP"
    body = f"<h2>Password Reset</h2><p>Your OTP is: <b>{otp}</b></p><p>Valid for 5 minutes.</p>"
    from fastapi.concurrency import run_in_threadpool
    await run_in_threadpool(send_email, email, subject, body)

# ---------------- User Routes ----------------
@user_router.post("/register")
async def register_user(new_user: User, background_tasks: BackgroundTasks):
    email = new_user.email.strip().lower()
    if collection.find_one({"username": new_user.username}):
        raise HTTPException(status_code=400, detail="Username already taken")
    if collection.find_one({"email": email}):
        raise HTTPException(status_code=400, detail="Email already registered")

    user_dict = new_user.dict()
    user_dict["email"] = email
    user_dict["password"] = pwd_context.hash(new_user.password)
    user_dict["created_at"] = datetime.utcnow()
    resp = collection.insert_one(user_dict)

    background_tasks.add_task(send_welcome_email, email, new_user.full_name)
    return {"status": "success", "id": str(resp.inserted_id), "message": "User registered successfully"}

@user_router.post("/signin")
async def signin_user(login_user: LoginUser):
    email = login_user.email.strip().lower()
    user_in_db = collection.find_one({"email": email})

    if not user_in_db or not pwd_context.verify(login_user.password, user_in_db["password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    return {
        "status": "success",
        "message": "Login successful",
        "user": {
            "full_name": user_in_db.get("full_name", ""),
            "username": user_in_db.get("username", ""),
            "email": user_in_db["email"]
        }
    }

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

@user_router.post("/reset-password")
async def reset_password(request: ResetPasswordRequest):
    email = request.email.strip().lower()
    user = collection.find_one({"email": email})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    otp_in_db = user.get("reset_otp")
    expiry_in_db = user.get("reset_expiry")
    if not otp_in_db or not expiry_in_db:
        raise HTTPException(status_code=400, detail="OTP not generated or already used")

    if isinstance(expiry_in_db, str):
        expiry_in_db = datetime.fromisoformat(expiry_in_db)
    if datetime.utcnow() > expiry_in_db:
        raise HTTPException(status_code=400, detail="OTP expired")
    if request.otp != otp_in_db:
        raise HTTPException(status_code=400, detail="Invalid OTP")

    hashed_password = pwd_context.hash(request.new_password)
    collection.update_one(
        {"email": email},
        {"$set": {"password": hashed_password}, "$unset": {"reset_otp": "", "reset_expiry": ""}}
    )
    return {"status": "success", "message": "Password reset successfully"}

@user_router.delete("/delete/{email}")
async def delete_user(email: str):
    email = email.strip().lower()
    result = collection.delete_one({"email": email})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
    return {"status": "success", "message": "Account deleted"}

# ---------------- News Functions ----------------
def fetch_and_store_news(lang="en", pages=2):
    if not NEWSDATA_API_KEY:
        print("❌ No API key found for NewsData.io")
        return

    url = f"https://newsdata.io/api/1/news?apikey={NEWSDATA_API_KEY}&country=in&language={lang}"
    page_count = 0
    inserted_total = 0

    while url and page_count < pages:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                print(f"[{lang}] Failed: {response.status_code}")
                break

            data = response.json()
            articles = data.get("results", [])

            for a in articles:
                link = a.get("link")
                if not link:
                    continue

                description = a.get("description", "") or ""
                if len(description) > 150:
                    description = description[:150].rstrip() + "..."

                doc = {
                    "title": a.get("title", ""),
                    "description": description,
                    "url": link,
                    "image": a.get("image_url", ""),
                    "publishedAt": a.get("pubDate", ""),
                    "language": lang,
                    "source": "NewsData.io",
                    "createdAt": datetime.utcnow()
                }

                # ✅ Upsert to avoid duplicates
                result = news_collection.update_one(
                    {"url": link},
                    {"$setOnInsert": doc},
                    upsert=True
                )

                if result.upserted_id:
                    inserted_total += 1

            next_page = data.get("nextPage")
            if next_page:
                url = f"https://newsdata.io/api/1/news?apikey={NEWSDATA_API_KEY}&country=in&language={lang}&page={next_page}"
                page_count += 1
            else:
                break

        except Exception as e:
            print(f"[{lang}] Error: {e}")
            break

    print(f"[{lang}] ✅ Inserted {inserted_total} new news articles")

def cleanup_old_news():
    one_week_ago = datetime.utcnow() - timedelta(days=7)
    result = news_collection.delete_many({"createdAt": {"$lt": one_week_ago}})
    print(f"🧹 Deleted {result.deleted_count} old news articles")
    return result.deleted_count

# ---------------- News Routes ----------------
@news_router.get("/")
def get_news(language: str = "en", limit: int = 20):
    news = list(news_collection.find({"language": language}).sort("createdAt", -1).limit(limit))
    for n in news:
        n["_id"] = str(n["_id"])
    return {"articles": news}

@news_router.get("/all")
def get_all_news():
    news = list(news_collection.find().sort("createdAt", -1))
    for n in news:
        n["_id"] = str(n["_id"])
    return {"count": len(news), "articles": news}

@news_router.get("/refresh")
def refresh_news(secret: str = Query(...)):
    if secret != CRON_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    fetch_and_store_news("en")
    fetch_and_store_news("hi")
    return {"status": "success", "message": "News refreshed"}

# ---------------- Include Routers ----------------
app.include_router(user_router)
app.include_router(news_router)
