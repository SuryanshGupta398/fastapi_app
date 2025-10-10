import os
import requests
from datetime import datetime, timedelta
import joblib
import numpy as np
from fastapi import FastAPI, APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from fastapi.concurrency import run_in_threadpool
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------- Local imports ----------------
from configuration import collection, news_collection
from models import User, LoginUser
from gmail_service import send_email

# ---------------- FastAPI setup ----------------
app = FastAPI()
user_router = APIRouter(prefix="/users", tags=["Users"])
news_router = APIRouter(prefix="/news", tags=["News"])

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ---------------- Load environment ----------------
NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY")
CRON_SECRET = os.getenv("CRON_SECRET")

# ---------------- Load ML model ----------------
MODEL_PATH = "full_news_model.pkl"
VECTORIZER_PATH = "full_tfidf_vectorizer.pkl"
ENCODER_PATH = "label_encoder1.pkl"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
label_encoder = joblib.load(ENCODER_PATH)

print("✅ ML Model, Vectorizer & Encoder loaded successfully!")

# ---------------- Classes ----------------
ALL_CLASSES = ['Business', 'Crime', 'Entertainment', 'Food', 'Science', 'Sports', 'International', 'Other']
# Ensure label encoder is synced with ALL_CLASSES
label_encoder.fit(ALL_CLASSES)

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
    await run_in_threadpool(send_email, email, subject, body)

async def send_otp_email(email: str, otp: str):
    subject = "Password Reset OTP"
    body = f"<h2>Password Reset</h2><p>Your OTP is: <b>{otp}</b></p><p>Valid for 5 minutes.</p>"
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

# ---------------- News Fetch & Train ----------------
def fetch_and_store_news(lang="en", pages=2):
    if not NEWSDATA_API_KEY:
        print("❌ No API key found for NewsData.io")
        return

    url = f"https://newsdata.io/api/1/news?apikey={NEWSDATA_API_KEY}&country=in&language={lang}"
    page_count = 0
    inserted_total = 0
    X_new, y_new_str = [], []

    while url and page_count < pages:
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

            title = a.get("title", "")
            if not title.strip():
                continue

            desc = a.get("description", "") or ""
            if len(desc) > 150:
                desc = desc[:150].rstrip() + "..."

            # Predict category using current model
            X_vec = vectorizer.transform([title])
            y_pred = model.predict(X_vec)
            try:
                category = label_encoder.inverse_transform(y_pred)[0]
            except ValueError:
                print("⚠️ Unseen label predicted:", y_pred)
                category = "Other"

            # Save news in DB
            doc = {
                "title": title,
                "description": desc,
                "url": link,
                "image": a.get("image_url", ""),
                "publishedAt": a.get("pubDate", ""),
                "language": lang,
                "source": "NewsData.io",
                "category": category,
                "createdAt": datetime.utcnow()
            }

            result = news_collection.update_one({"url": link}, {"$setOnInsert": doc}, upsert=True)
            if result.upserted_id:
                inserted_total += 1
                X_new.append(title)
                y_new_str.append(category)

        next_page = data.get("nextPage")
        if next_page:
            url = f"https://newsdata.io/api/1/news?apikey={NEWSDATA_API_KEY}&country=in&language={lang}&page={next_page}"
            page_count += 1
        else:
            break

    print(f"[{lang}] ✅ Inserted {inserted_total} new articles")

    # Incrementally train the model
    if X_new:
        X_vec_new = vectorizer.transform(X_new)
        y_new_int = label_encoder.transform(y_new_str)  # convert strings to integers
        all_classes_int = np.arange(len(label_encoder.classes_))
        model.partial_fit(X_vec_new, y_new_int, classes=all_classes_int)
        joblib.dump(model, MODEL_PATH)
        print(f"🤖 Model improved with {len(X_new)} new samples!")

def cleanup_old_news():
    one_week_ago = datetime.utcnow() - timedelta(days=7)
    result = news_collection.delete_many({"createdAt": {"$lt": one_week_ago}})
    print(f"🧹 Deleted {result.deleted_count} old news articles")

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

@news_router.get("/category/{category}")
def get_news_by_category(
    category: str,
    language: str = "en",
    limit: int = 50
):
    # Filter news by category and language
    news = list(
        news_collection.find({"category": category, "language": language})
        .sort("createdAt", -1)
        .limit(limit)
    )
    for n in news:
        n["_id"] = str(n["_id"])
    return {"count": len(news), "articles": news}

@news_router.get("/refresh")
def refresh_news(secret: str = Query(...)):
    if secret != CRON_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    fetch_and_store_news("en")
    fetch_and_store_news("hi")
    cleanup_old_news()

    # Calculate accuracy after refresh
    news_docs = list(news_collection.find())
    X_test = [doc["title"] for doc in news_docs]
    y_true_str = [doc.get("category", "Other") for doc in news_docs]
    X_vec = vectorizer.transform(X_test)
    y_true_int = label_encoder.transform(y_true_str)
    y_pred_int = model.predict(X_vec)
    accuracy = round(accuracy_score(y_true_int, y_pred_int) * 100, 2)

    return {
        "status": "success",
        "message": "News fetched & model improved",
        "accuracy": accuracy
    }

# ---------------- Register Routers ----------------
app.include_router(user_router)
app.include_router(news_router)
