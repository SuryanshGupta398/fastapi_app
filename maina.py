import os
import random
import requests
from datetime import datetime, timedelta

from fastapi import FastAPI, APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig, MessageType
from apscheduler.schedulers.background import BackgroundScheduler

from configuration import collection, news_collection  # MongoDB collections
from models import User, LoginUser

# ------------------ App & Routers ------------------
app = FastAPI()
user_router = APIRouter(prefix="/users", tags=["Users"])
news_router = APIRouter(prefix="/news", tags=["News"])

# ------------------ Password Hashing ------------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ------------------ GNews API Key ------------------
NEWSDATA_API_KEY = os.getenv("GNEWS_KEY")

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

# ------------------ Cron Secret ------------------
CRON_SECRET = os.getenv("CRON_SECRET")  # change in Render env

# ------------------ Request Models ------------------
class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    email: EmailStr
    otp: str
    new_password: str

# ------------------ Health Endpoint ------------------
@app.api_route("/health", methods=["GET", "HEAD"])
def health_check():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}

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
    if isinstance(expiry_in_db, str):
        expiry_in_db = datetime.fromisoformat(expiry_in_db) if expiry_in_db else None

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
    if isinstance(expiry_in_db, str):
        expiry_in_db = datetime.fromisoformat(expiry_in_db)
    now = datetime.utcnow()

    if not otp_in_db or not expiry_in_db:
        raise HTTPException(status_code=400, detail="OTP not generated. Request a new one.")
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
def fetch_and_store_news(lang="en", pages=2):
    url = f"https://newsdata.io/api/1/news?apikey={NEWSDATA_API_KEY}&country=in&language={lang}"
    page_count = 0
    inserted_total = 0

    while url and page_count < pages:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                print(f"[{lang}] Failed: {response.status_code} {response.text}")
                break

            data = response.json()
            articles = data.get("results", [])
            print(f"[{lang}] Page {page_count+1}: {len(articles)} articles")

            for a in articles:
                link = a.get("link")
                if not link:
                    print("⚠️ Skipping article without link:", a)
                    continue

                if news_collection.count_documents({"url": link}) == 0:
                    doc = {
                        "title": a.get("title", ""),
                        "description": a.get("description", ""),
                        "url": link,
                        "image": a.get("image_url", ""),
                        "publishedAt": a.get("pubDate", ""),
                        "language": lang,
                        "source": "NewsData.io",
                        "createdAt": datetime.utcnow()
                    }
                    try:
                        news_collection.insert_one(doc)
                        inserted_total += 1
                    except Exception as e:
                        print("❌ Insert failed:", e)

            print(f"[{lang}] Inserted {inserted_total} articles so far")

            # go to next page if available
            next_page = data.get("nextPage")
            if next_page:
                url = f"https://newsdata.io/api/1/news?apikey={NEWSDATA_API_KEY}&country=in&language={lang}&page={next_page}"
                page_count += 1
            else:
                break

        except Exception as e:
            print(f"[{lang}] Error: {e}")
            break

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

@news_router.get("/all")
def get_all_news():
    news = list(news_collection.find().sort("createdAt", -1))  # get all news, latest first
    for n in news:
        n["_id"] = str(n["_id"])  # convert ObjectId to string
    return {"count": len(news), "articles": news}

# ✅ Allow both GET and POST for refresh
@news_router.api_route("/refresh", methods=["GET", "POST"])
def refresh_news(secret: str = Query(..., description="Cron secret for security")):
    if secret != CRON_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")
    fetch_and_store_news("en")
    fetch_and_store_news("hi")
    return {"status": "success", "message": "News refreshed"}
# ------------------ Scheduler ------------------
scheduler = BackgroundScheduler()

# Automatic refresh every 30 minutes
scheduler.add_job(lambda: fetch_and_store_news("en"), 'interval', minutes=30, id="refresh_en_news")
scheduler.add_job(lambda: fetch_and_store_news("hi"), 'interval', minutes=30, id="refresh_hi_news")

# Automatic cleanup every 24 hours
scheduler.add_job(cleanup_old_news, 'interval', hours=24, id="cleanup_old_news")

scheduler.start()
print("Scheduler started: fetching news every 30 minutes and cleaning up every 24 hours.")

# ------------------ Include Routers ------------------
app.include_router(user_router)
app.include_router(news_router)
