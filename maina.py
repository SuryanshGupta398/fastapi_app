import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import random
import joblib
import numpy as np
from fastapi import FastAPI, APIRouter, HTTPException, BackgroundTasks, Query, UploadFile, File, Form
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from fastapi.concurrency import run_in_threadpool
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# ---------------- Local imports ----------------
from configuration import collection, news_collection
from models import User, LoginUser
from gmail_service import send_email

# ---------------- FastAPI setup ----------------
app = FastAPI()
user_router = APIRouter(prefix="/users", tags=["Users"])
news_router = APIRouter(prefix="/news", tags=["News"])
report_router = APIRouter(prefix="/report", tags=["Report"])

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ---------------- Load environment ----------------
NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY")
CRON_SECRET = os.getenv("CRON_SECRET")

# ---------------- ML model paths ----------------
MODEL_PATH = "full_news_model.pkl"
VECTORIZER_PATH = "full_tfidf_vectorizer.pkl"
ENCODER_PATH = "label_encoder1.pkl"

# ---------------- Classes ----------------
ALL_CLASSES = [
    'Business', 'Crime', 'Entertainment', 'Food', 'Science',
    'Sports', 'International', 'Other', 'Health', 'Politics'
]

print("✅ ML Model, Vectorizer & Encoder ready!")

# ---------------- Keyword overrides ----------------
CATEGORY_KEYWORDS = {
    "Business": ["company", "startup", "brand", "market", "investment", "IPO", "business", "deal", "corporate", "firm"],
    "Sports": ["match", "tournament", "football", "cricket", "goal", "player", "league", "score"],
    "Entertainment": ["movie", "film", "celebrity", "song", "album", "show", "series", "tv"],
    "Food": ["restaurant", "recipe", "dish", "cuisine", "menu", "food", "chef"],
    "Science": ["research", "experiment", "discovery", "scientist"],
    "Health": ["disease", "medicine", "vaccine", "hospital", "covid", "health"],
    "Politics": ["election", "government", "minister", "policy", "vote"]
}

def categorize_with_keywords(text: str, predicted: str) -> str:
    text_lower = text.lower()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                return cat
    return predicted


# ---------------- SAFE MODEL LOADING FIX ----------------
model = None
vectorizer = None
label_encoder = None
current_accuracy = 0.0

def _safe_load_joblib(path, desc):
    try:
        obj = joblib.load(path)
        print(f"✅ Loaded {desc} from {path}")
        return obj
    except FileNotFoundError:
        print(f"⚠️ {desc} not found at {path}. Creating fallback.")
    except Exception as e:
        print(f"⚠️ Error loading {desc}: {e}")
    return None

# Load vectorizer
vectorizer = _safe_load_joblib(VECTORIZER_PATH, "TF-IDF Vectorizer")
if vectorizer is None:
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=5000)
    print("ℹ️ Created fallback TfidfVectorizer (unfitted).")

# Load encoder
label_encoder = _safe_load_joblib(ENCODER_PATH, "Label Encoder")
if label_encoder is None:
    label_encoder = LabelEncoder()
    label_encoder.fit(ALL_CLASSES)
    print("ℹ️ Fitted fallback LabelEncoder using ALL_CLASSES.")

# Load model
model = _safe_load_joblib(MODEL_PATH, "News Classifier Model")
if model is None:
    model = SGDClassifier(max_iter=1000, tol=1e-3)
    print("ℹ️ Created fallback SGDClassifier model.")

if not NEWSDATA_API_KEY:
    print("⚠️ NEWSDATA_API_KEY not set. fetch_and_store_news() will fail if called.")


# ---------------- Request Models ----------------
class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    email: EmailStr
    otp: str
    new_password: str

# ---------------- Health Check ----------------
@app.get("/health")
def health_check():
    return {"status": "ok", "time": datetime.utcnow().isoformat(), "model_accuracy": current_accuracy}

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

otp_store = {}

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

    background_tasks.add_task(lambda: send_welcome_email(email, new_user.full_name))
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

# ---------------- Forgot Password / Reset ----------------
@user_router.post("/forgot-password")
async def forgot_password(request: ForgotPasswordRequest, background_tasks: BackgroundTasks):
    user = collection.find_one({"email": request.email.lower()})
    if not user:
        raise HTTPException(status_code=404, detail="Email not registered")
    otp = str(random.randint(100000, 999999))
    otp_store[request.email.lower()] = {"otp": otp, "expires": datetime.utcnow() + timedelta(minutes=5)}
    background_tasks.add_task(lambda: send_otp_email(request.email, otp))
    return {"status": "success", "message": "OTP sent successfully"}

@user_router.post("/reset-password")
async def reset_password(request: ResetPasswordRequest):
    record = otp_store.get(request.email.lower())
    if not record or record["otp"] != request.otp:
        raise HTTPException(status_code=400, detail="Invalid OTP")
    if datetime.utcnow() > record["expires"]:
        raise HTTPException(status_code=400, detail="OTP expired")

    hashed_pwd = pwd_context.hash(request.new_password)
    result = collection.update_one({"email": request.email.lower()}, {"$set": {"password": hashed_pwd}})
    if result.modified_count == 0:
        raise HTTPException(status_code=400, detail="Password update failed")
    del otp_store[request.email.lower()]
    return {"status": "success", "message": "Password reset successful"}

# ---------------- Delete Account ----------------
@user_router.delete("/delete-account")
async def delete_account(email: EmailStr, password: str):
    user = collection.find_one({"email": email.lower()})
    if not user or not pwd_context.verify(password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    collection.delete_one({"email": email.lower()})
    return {"status": "success", "message": "Account deleted successfully"}

# ---------------- News Fetch & Train ----------------
news_collection.create_index("url", unique=True)

def fetch_and_store_news(lang="en", pages=2):
    url = f"https://newsdata.io/api/1/news?apikey={NEWSDATA_API_KEY}&country=in&language={lang}"
    page_count, inserted_total = 0, 0
    X_new, y_new_str = [], []

    while url and page_count < pages:
        resp = requests.get(url, timeout=50)
        if resp.status_code != 200:
            break
        data = resp.json()
        for a in data.get("results", []):
            title = a.get("title", "")
            if not title:
                continue
            desc = a.get("description", "") or ""
            X_vec = vectorizer.transform([title])
            y_pred = model.predict(X_vec)
            category = label_encoder.inverse_transform(y_pred)[0]
            category = categorize_with_keywords(title, category)
            doc = {
                "title": title,
                "description": desc[:150],
                "url": a.get("link", ""),
                "image": a.get("image_url", ""),
                "publishedAt": a.get("pubDate", ""),
                "language": lang,
                "source": "NewsData.io",
                "category": category,
                "createdAt": datetime.utcnow()
            }
            try:
                news_collection.insert_one(doc)
                inserted_total += 1
                X_new.append(title)
                y_new_str.append(category)
            except Exception as e:
                if "duplicate key" not in str(e).lower():
                    print("⚠️ Insert error:", e)
        next_page = data.get("nextPage")
        if next_page:
            url = f"https://newsdata.io/api/1/news?apikey={NEWSDATA_API_KEY}&country=in&language={lang}&page={next_page}"
            page_count += 1
        else:
            break
    print(f"[{lang}] ✅ Inserted {inserted_total} new articles")
    if X_new:
        X_vec_new = vectorizer.transform(X_new)
        y_new_int = label_encoder.transform(y_new_str)
        all_classes_int = np.arange(len(label_encoder.classes_))
        model.partial_fit(X_vec_new, y_new_int, classes=all_classes_int)
        joblib.dump(model, MODEL_PATH)
        print(f"🤖 Model improved with {len(X_new)} new samples!")

# ---------------- News Routes ----------------
@news_router.get("/")
def get_news(language: str = "en", limit: int = 20):
    news = list(news_collection.find({"language": language}).sort("createdAt", -1).limit(limit))
    for n in news:
        n["_id"] = str(n["_id"])
    return {"articles": news}

@news_router.get("/category/{category}")
def get_news_by_category(category: str, language: str = "en", limit: int = 50):
    news = list(news_collection.find({"category": category, "language": language}).sort("createdAt", -1).limit(limit))
    for n in news:
        n["_id"] = str(n["_id"])
    return {"count": len(news), "articles": news}

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

    news_docs = list(news_collection.find({"category": {"$exists": True}}))
    if news_docs:
        X_test = [doc["title"] for doc in news_docs]
        y_true_str = [doc["category"] for doc in news_docs]
        X_vec = vectorizer.transform(X_test)
        y_true_int = label_encoder.transform(y_true_str)
        y_pred_int = model.predict(X_vec)
        accuracy = round(accuracy_score(y_true_int, y_pred_int) * 100, 2)
    else:
        accuracy = 0.0

    global current_accuracy
    current_accuracy = accuracy

    return {"status": "success", "message": "News fetched & model improved", "accuracy": accuracy}

TRENDING_KEYWORDS = [
    "breaking", "exclusive", "update", "live", "urgent", "just in", "latest", "alert"
]

@news_router.get("/trending-smart")
def get_smart_trending_news(limit: int = 100):
    try:
        now = datetime.utcnow()
        last_7_days = now - timedelta(days=7)
        recent_news = list(news_collection.find({"createdAt": {"$gte": last_7_days}}).limit(300))

        trending = []
        for n in recent_news:
            views = n.get("views", 0)
            title = n.get("title", "").lower()
            created_at = n.get("createdAt", now)
            hours_old = (now - created_at).total_seconds() / 3600

            recency_boost = max(0, int(168 - hours_old)) // 8
            keyword_boost = 20 if any(kw in title for kw in TRENDING_KEYWORDS) else 0
            score = (views * 2.5) + keyword_boost + recency_boost

            n["trending_score"] = score
            n["_id"] = str(n["_id"])
            trending.append(n)

        trending = sorted(trending, key=lambda x: x["trending_score"], reverse=True)[:limit]
        return {"status": "success", "count": len(trending), "articles": trending}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching smart trending news: {str(e)}")

# ---------------- Report Route ----------------
@report_router.post("/misinformation")
async def report_misinformation(
    email: str = Form(...),
    link: str = Form(""),
    reason: str = Form(...),
    proof: UploadFile = File(None)
):
    try:
        attachment_path = None
        if proof:
            attachment_path = f"temp_{proof.filename}"
            with open(attachment_path, "wb") as f:
                f.write(await proof.read())

        subject_admin = "🚨 New Misinformation Report"
        body_admin = f"""
        <h2>New Misinformation Report</h2>
        <p><b>Reporter Email:</b> {email}</p>
        <p><b>News Link:</b> {link or 'No link provided'}</p>
        <p><b>Reason:</b> {reason}</p>
        <p><i>🕓 Reported at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</i></p>
        """

        send_email(
            to_email=os.getenv("MAIL_USERNAME"),
            subject=subject_admin,
            body=body_admin,
            attachment_path=attachment_path
        )

        subject_user = "✅ Thanks for Reporting Misinformation!"
        body_user = f"""
        <h3>Hi there,</h3>
        <p>Thank you for helping us fight misinformation!</p>
        <p>We’ll review your report and take appropriate action.</p>
        <br>
        <p>— The Fake News Detector Team</p>
        """

        send_email(to_email=email, subject=subject_user, body=body_user)

        if attachment_path and os.path.exists(attachment_path):
            os.remove(attachment_path)

        return {"status": "success", "message": "Report submitted successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing report: {str(e)}")

# ---------------- Register Routers ----------------
app.include_router(user_router)
app.include_router(news_router)
app.include_router(report_router)
