import os
import re
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
from pymongo import DESCENDING 
from sentence_transformers import SentenceTransformer, util

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
semantic_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ---------------- Load environment ----------------
NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY")
CRON_SECRET = os.getenv("CRON_SECRET")
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")

# ---------------- ML model paths ----------------
MODEL_PATH = "full_news_model.pkl"
VECTORIZER_PATH = "full_tfidf_vectorizer.pkl"
ENCODER_PATH = "label_encoder1.pkl"
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

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

FAKE_NEWS_PATTERNS = [
    # Clickbait patterns
    "you won't believe", "shocked the world", "doctors hate", "this one trick",
    "miracle cure", "secret they don't want", "instant results", "viral secret",
    "everyone is talking about", "celebrities are", "shocking news", 
    "the truth about", "they don't want you to know", "mainstream media won't tell",
    
    # Sensational patterns
    "breaking.*shocking", "urgent.*warning", "alert.*emergency", "crisis.*now",
    "exposed.*truth", "scandal.*revealed", "cover.up", "conspiracy",
    
    # Financial scams
    "make money fast", "earn.*from home", "get rich quick", "instant cash",
    "free money", "guaranteed profit", "investment secret",
    
    # Health scams
    "lose weight fast", "burn fat instantly", "cure.*overnight", "medical breakthrough they hide",
    
    # Emotional manipulation
    "will make you cry", "heartbreaking", "tears of joy", "you need to see this"
]

CREDIBLE_INDICATORS = [
    "reuters", "associated press", "ap news", "bbc", "cnn", "al jazeera", 
    "official statement", "government report", "study shows", "research indicates",
    "according to data", "statistics show", "peer-reviewed", "confirmed", "verified"
]

TRUSTED_DOMAINS = [
    "reuters.com", "ap.org", "bbc.com", "bbc.co.uk", "aljazeera.com",
    "nytimes.com", "washingtonpost.com", "theguardian.com"
]

def get_semantic_embedding(text: str):
    return semantic_model.encode(text, convert_to_tensor=True)

def better_similarity(a: str, b: str) -> float:
    a_words = set(a.split())
    b_words = set(b.split())

    if not a_words or not b_words:
        return 0

    overlap = a_words.intersection(b_words)
    score = len(overlap) / max(len(a_words), len(b_words))

    return score

def ultra_fast_pattern_check(headline: str) -> dict:
    """Ultra-fast pattern matching"""
    headline_lower = headline.lower().strip()
    
    # Check for fake news patterns
    fake_score = 0
    for pattern in FAKE_NEWS_PATTERNS:
        if re.search(pattern, headline_lower):
            fake_score += 1
    
    # Check for credible indicators
    credible_score = 0
    for indicator in CREDIBLE_INDICATORS:
        if indicator in headline_lower:
            credible_score += 1
    
    # Text analysis
    words = headline.split()
    word_count = len(words)
    has_caps = any(word.isupper() for word in words if len(word) > 3)
    has_exclamation = '!' in headline
    
    # Calculate confidence
    if fake_score >= 2:
        confidence = min(0.3 + (fake_score * 0.15), 0.9)
        rating = "Fake"
        reason = f"Multiple fake news patterns detected"
    elif credible_score >= 2:
        confidence = min(0.7 + (credible_score * 0.1), 0.9)
        rating = "True" 
        reason = f"Multiple credible indicators found"
    elif credible_score == 1:
        confidence = 0.6
        rating = "Likely True"
        reason = "Contains credible reference"
    elif fake_score == 1:
        confidence = 0.4
        rating = "Suspicious"
        reason = "One fake news pattern detected"
    else:
        confidence = 0.5
        rating = "Uncertain"
        reason = "No clear indicators found"
    
    # Adjust for sensationalism
    if has_exclamation and has_caps and word_count < 10:
        confidence = max(confidence - 0.2, 0.1)
        if rating != "Fake":
            rating = "Sensational"
        reason = "Uses sensational language"
    
    return {
        "rating": rating,
        "confidence": round(confidence, 2),
        "reason": reason,
        "fake_patterns": fake_score,
        "credible_indicators": credible_score
    }

def fast_mongodb_check(headline: str) -> dict:
    """Fast MongoDB check with improved fuzzy matching + safer recent filter"""
    try:
        headline = headline.lower()

        # Fetch last 15 days (safe range) — not just 3 days
        recent_limit = datetime.utcnow() - timedelta(days=15)

        recent_news = list(news_collection.find({
            "createdAt": {"$gte": recent_limit}
        }).limit(200))

        # If still empty, fallback: remove date filter entirely
        if not recent_news:
            recent_news = list(news_collection.find().limit(200))

        matches = []

        for news in recent_news:
            news_title = news.get("title", "").lower()
            if not news_title:
                continue

            # Use better overlap scoring
            similarity = better_similarity(headline, news_title)

            if similarity >= 0.2:  # Lower threshold
                matches.append({
                    "title": news.get("title", ""),
                    "similarity": round(similarity, 2),
                    "url": news.get("url", ""),
                    "source": news.get("source", "Unknown"),
                })

        matches.sort(key=lambda x: x["similarity"], reverse=True)
        top_matches = matches[:5]

        return {
            "found": len(top_matches) > 0,
            "count": len(top_matches),
            "matches": top_matches
        }

    except Exception as e:
        print("MongoDB check error:", e)
        return {"found": False, "count": 0, "matches": []}

def semantic_mongo_search(headline: str, top_k: int = 5):
    query_emb = semantic_model.encode(headline, convert_to_tensor=True)

    # Fetch last 15 days news
    recent_limit = datetime.utcnow() - timedelta(days=15)
    docs = list(news_collection.find(
        {"embedding": {"$exists": True},
         "createdAt": {"$gte": recent_limit}},
        {"title": 1, "url": 1, "source": 1, "embedding": 1}
    ))

    if not docs:
        docs = list(news_collection.find({"embedding": {"$exists": True}}))

    # Compute similarity
    scores = []
    for d in docs:
        emb = np.array(d["embedding"])
        sim = util.cos_sim(query_emb, emb).item()
        scores.append({
            "title": d.get("title", ""),
            "url": d.get("url", ""),
            "source": d.get("source", "Unknown"),
            "similarity": round(float(sim), 3)
        })

    scores.sort(key=lambda x: x["similarity"], reverse=True)
    return scores[:top_k]

def fast_gnews_check(headline: str) -> dict:
    """Fast GNews check with timeout"""
    GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")
    if not GNEWS_API_KEY:
        return {"error": "API key missing"}
    
    try:
        # Use only first 4 words for search (faster)
        search_terms = " ".join(headline.split()[:4])
        url = f"https://gnews.io/api/v4/search?q={search_terms}&token={GNEWS_API_KEY}&lang=en&max=2"
        
        response = requests.get(url, timeout=3)  # 3 second timeout
        
        if response.status_code == 200:
            data = response.json()
            if "articles" in data and data["articles"]:
                articles = []
                for article in data["articles"][:2]:
                    # Check if domain is trusted
                    domain_trusted = any(domain in article.get("url", "") for domain in TRUSTED_DOMAINS)
                    articles.append({
                        "title": article["title"],
                        "url": article["url"],
                        "source": article.get("source", {}).get("name", "Unknown"),
                        "trusted_domain": domain_trusted
                    })
                
                return {
                    "found": True,
                    "count": len(articles),
                    "articles": articles,
                    "trusted_sources": sum(1 for a in articles if a["trusted_domain"])
                }
        return {"found": False, "count": 0}
    except requests.exceptions.Timeout:
        return {"error": "Timeout"}
    except Exception as e:
        return {"error": str(e)}

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

def remove_duplicates(articles):
    """Remove duplicate articles by URL or similar title."""
    seen_urls = set()
    seen_titles = set()
    unique_articles = []
    for a in articles:
        url = a.get("url", "")
        title = a.get("title", "").lower().strip()
        if not title:
            continue
        if url in seen_urls or title in seen_titles:
            continue
        seen_urls.add(url)
        seen_titles.add(title)
        unique_articles.append(a)
    return unique_articles

def fetch_and_store_news(lang="en", pages=2):
    """
    Fetch news from both NewsData.io and GNews, merge, deduplicate, and store.
    """
    NEWS_API_KEY = os.getenv("NEWSDATA_API_KEY")
    GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")
    inserted_total = 0
    X_new, y_new_str = [], []

    all_articles = []

    # ---------------- Fetch from NewsData.io ----------------
    newsdata_url = f"https://newsdata.io/api/1/news?apikey={NEWS_API_KEY}&country=in&language={lang}"
    page_count = 0
    while newsdata_url and page_count < pages:
        resp = requests.get(newsdata_url, timeout=50)
        if resp.status_code != 200:
            break
        data = resp.json()
        for a in data.get("results", []):
            title = a.get("title", "")
            if not title:
                continue
            desc = a.get("description", "") or ""
            all_articles.append({
                "title": title,
                "description": desc[:150],
                "url": a.get("link", ""),
                "image": a.get("image_url", ""),
                "publishedAt": a.get("pubDate", ""),
                "language": lang,
                "source": "NewsData.io"
            })
        next_page = data.get("nextPage")
        if next_page:
            newsdata_url = f"https://newsdata.io/api/1/news?apikey={NEWS_API_KEY}&country=in&language={lang}&page={next_page}"
            page_count += 1
        else:
            break

    # ---------------- Fetch from GNews ----------------
    try:
        gnews_url = f"https://gnews.io/api/v4/top-headlines?lang={lang}&country=in&max=50&apikey={GNEWS_API_KEY}"
        g_resp = requests.get(gnews_url, timeout=50)
        if g_resp.status_code == 200:
            g_data = g_resp.json()
            for a in g_data.get("articles", []):
                all_articles.append({
                    "title": a.get("title", ""),
                    "description": (a.get("description") or "")[:150],
                    "url": a.get("url", ""),
                    "image": a.get("image", ""),
                    "publishedAt": a.get("publishedAt", ""),
                    "language": lang,
                    "source": "GNews"
                })
    except Exception as e:
        print("⚠️ Error fetching from GNews:", e)

    # ---------------- Deduplicate ----------------
    unique_articles = remove_duplicates(all_articles)
    print(f"🧩 Combined {len(all_articles)} articles → {len(unique_articles)} unique after deduplication")

    # ---------------- Store in DB ----------------
    for a in unique_articles:
        title = a["title"]
        X_vec = vectorizer.transform([title])
        y_pred = model.predict(X_vec)
        category = label_encoder.inverse_transform(y_pred)[0]
        category = categorize_with_keywords(title, category)
        embedding = semantic_model.encode(title).tolist()
        doc = {
            **a,
            "category": category,
            "createdAt": datetime.utcnow(),
            "embedding": embedding           # <--- store semantic vector
        }

        try:
            news_collection.insert_one(doc)
            inserted_total += 1
            X_new.append(title)
            y_new_str.append(category)
        except Exception as e:
            if "duplicate key" not in str(e).lower():
                print("⚠️ Insert error:", e)

    print(f"[{lang}] ✅ Inserted {inserted_total} new articles after deduplication")

    # ---------------- Retrain model ----------------
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

news_collection.create_index([("createdAt", DESCENDING)])

@news_router.get("/all")
def get_all_news(limit: int = 1000):
    news = list(news_collection.find().sort("createdAt", DESCENDING).limit(limit))
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

        # Fetch recent 7-day news, focusing on potentially trending topics
        recent_news = list(
            news_collection.find(
                {"createdAt": {"$gte": last_7_days}},
                {"_id": 1, "title": 1, "description": 1, "views": 1,
                 "shares": 1, "likes": 1, "category": 1,
                 "createdAt": 1, "image_url": 1, "source": 1}
            ).limit(400)
        )

        trending = []
        for n in recent_news:
            views = n.get("views", 0)
            shares = n.get("shares", 0)
            likes = n.get("likes", 0)
            title = n.get("title", "").lower()
            created_at = n.get("createdAt", now)

            # Hours since publication
            hours_old = (now - created_at).total_seconds() / 3600

            # Recency boost (fresh content gets more weight)
            recency_boost = max(0, 150 - hours_old) / 10

            # Keyword importance
            keyword_boost = 25 if any(kw in title for kw in TRENDING_KEYWORDS) else 0

            # Engagement weight — gives importance to viral stories
            engagement_score = (views * 1.5) + (shares * 3) + (likes * 2)

            # Combine to form smart trending score
            trending_score = engagement_score + recency_boost + keyword_boost

            n["trending_score"] = round(trending_score, 2)
            n["_id"] = str(n["_id"])
            trending.append(n)

        # Sort and limit
        trending = sorted(trending, key=lambda x: x["trending_score"], reverse=True)[:limit]

        # Optional: highlight top 5 with tag for UI
        for i, news in enumerate(trending[:5]):
            news["highlight"] = True

        return {
            "status": "success",
            "count": len(trending),
            "articles": trending
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching smart trending news: {str(e)}"
        )

# ---------------- Verify News Route (Google Fact Check API) ----------------
# ---------------- Verify News Route (Integrated: Google Fact Check + Local DB) ----------------
@news_router.post("/verify-news")
async def verify_news_comprehensive_all(headline: str = Form(...)):
    """
    Comprehensive news verification:
    - Pattern check
    - MongoDB exact/fuzzy match
    - MongoDB semantic search
    - GNews check
    - NewsData.io check
    """
    start_time = datetime.utcnow()
    headline = headline.strip()
    if not headline:
        raise HTTPException(status_code=400, detail="Headline required")

    # 1️⃣ Pattern check
    pattern_result = ultra_fast_pattern_check(headline)

    # 2️⃣ MongoDB exact/fuzzy check
    mongodb_result = fast_mongodb_check(headline)

    # 3️⃣ Semantic search in MongoDB
    semantic_results = semantic_mongo_search(headline, top_k=5)  # top 5 similar

    # 4️⃣ External GNews check
    gnews_result = fast_gnews_check(headline)

    # 5️⃣ NewsData.io check
    newsdata_results = []
    if NEWSDATA_API_KEY:
        try:
            search_terms = "+".join(headline.split()[:5])  # first 5 words
            url = f"https://newsdata.io/api/1/news?q={search_terms}&apikey={NEWSDATA_API_KEY}&language=en&country=in"
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                articles = data.get("results", [])[:3]  # top 3
                for a in articles:
                    newsdata_results.append({
                        "title": a.get("title"),
                        "description": a.get("description"),
                        "url": a.get("link"),
                        "source": a.get("source_id"),
                        "publishedAt": a.get("pubDate")
                    })
        except Exception as e:
            print("⚠️ NewsData.io check failed:", e)

    # 6️⃣ Calculate final confidence
    confidence_factors = [pattern_result["confidence"]]
    if mongodb_result.get("found"):
        db_boost = 0.15 + (mongodb_result["count"] * 0.05)
        confidence_factors.append(min(pattern_result["confidence"] + db_boost, 0.9))
    if semantic_results:
        semantic_boost = 0.2 + (len(semantic_results) * 0.05)
        confidence_factors.append(min(pattern_result["confidence"] + semantic_boost, 0.95))
    if gnews_result.get("found"):
        external_boost = 0.2 + (gnews_result["count"] * 0.08)
        if gnews_result.get("trusted_sources", 0) > 0:
            external_boost += 0.1
        confidence_factors.append(min(pattern_result["confidence"] + external_boost, 0.95))
    if newsdata_results:
        nd_boost = 0.1 + (len(newsdata_results) * 0.05)
        confidence_factors.append(min(pattern_result["confidence"] + nd_boost, 0.95))

    final_confidence = sum(confidence_factors) / len(confidence_factors)
    final_confidence = min(final_confidence, 0.95)

    # 7️⃣ Determine final rating
    if (gnews_result.get("found") or newsdata_results) and (mongodb_result.get("found") or semantic_results):
        final_rating = "Verified True"
    elif gnews_result.get("found") or semantic_results or mongodb_result.get("found") or newsdata_results:
        if pattern_result["rating"] == "Fake":
            final_rating = "Contradictory - verify manually"
        else:
            final_rating = "Likely True"
    else:
        final_rating = pattern_result["rating"]

    response_time = (datetime.utcnow() - start_time).total_seconds()

    return {
        "status": "success",
        "verified": final_rating in ["Verified True", "Likely True", "True"],
        "headline": headline,
        "rating": final_rating,
        "confidence": round(final_confidence, 2),
        "response_time_seconds": round(response_time, 2),
        "sources_checked": {
            "pattern_analysis": True,
            "database_matches": mongodb_result.get("count", 0),
            "semantic_matches": len(semantic_results),
            "gnews_matches": gnews_result.get("count", 0),
            "trusted_sources": gnews_result.get("trusted_sources", 0),
            "newsdata_matches": len(newsdata_results)
        },
        "evidence": {
            "pattern_result": pattern_result,
            "database_matches": mongodb_result.get("matches", []),
            "semantic_matches": semantic_results,
            "gnews_articles": gnews_result.get("articles", []),
            "newsdata_articles": newsdata_results
        },
        "message": f"Analysis completed in {response_time:.1f}s: {final_rating} ({final_confidence*100}% confidence)"
    }

# ---------------- Traveller Updates (Local DB only) ----------------
@news_router.get("/traveller-updates")
def traveller_updates(location: str = Query(..., description="City or country name")):
    """
    Fetches all news related to the given location from MongoDB.
    Includes travel and general updates — no external API used.
    """
    try:
        # Match location anywhere in title, description, category, or location fields
        location_regex = {"$regex": location, "$options": "i"}

        cursor = news_collection.find({
            "$or": [
                {"title": location_regex},
                {"description": location_regex},
                {"category": location_regex},
                {"location": location_regex}
            ]
        }).sort("createdAt", -1)

        results = list(cursor)
        for n in results:
            n["_id"] = str(n["_id"])

        if not results:
            return {
                "status": "not_found",
                "location": location,
                "verified": False,
                "confidence": 0.4,
                "count": 0,
                "all_news": [],
                "message": f"No news found for '{location}' in local database."
            }

        # Compute credibility/confidence based on count and sources
        credible_sources = list({n.get("source", "LocalDB") for n in results if n.get("source")})
        avg_confidence = 0.9 if len(results) > 5 else 0.75

        return {
            "status": "success",
            "location": location,
            "verified": True,
            "confidence": avg_confidence,
            "count": len(results),
            "credible_sources": credible_sources,
            "all_news": results,
            "message": f"Fetched {len(results)} news items for '{location}' from local DB."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching updates: {str(e)}")

# ---------------- Travel Route Updates ----------------
@news_router.get("/travel-route-updates")
def travel_route_updates(
    source: str = Query(..., description="Source city name"),
    destination: str = Query(..., description="Destination city name")
):
    """
    Fetches all news (not just travel-related) between source and destination cities.
    Includes any news where source, destination, or route cities appear in the title,
    description, category, or location fields. Uses local MongoDB only.
    """
    try:
        # 1️⃣ Define known route points (expandable dictionary)
        route_points = {
            "kanpur": ["Etawah", "Firozabad", "Agra", "Mathura", "Noida", "Delhi"],
            "mumbai": ["Surat", "Vadodara", "Udaipur", "Jaipur", "Gurugram", "Delhi"],
            "lucknow": ["Kanpur", "Agra", "Noida", "Delhi"]
        }

        src = source.lower()
        dest = destination.lower()

        # Get in-between route cities (if defined)
        route_cities = route_points.get(src, []) if dest in route_points.get(src, []) else []
        route_cities = [src, *route_cities, dest] if route_cities else [src, dest]

        # Create regex pattern for all route city names
        location_filter = {"$regex": "|".join(route_cities), "$options": "i"}

        # 2️⃣ Query MongoDB for any news containing those cities
        cursor = news_collection.find({
            "$or": [
                {"title": location_filter},
                {"description": location_filter},
                {"category": location_filter},
                {"location": location_filter}
            ]
        }).sort("createdAt", -1)

        results = list(cursor)
        for r in results:
            r["_id"] = str(r["_id"])

        # 3️⃣ Handle no results
        if not results:
            return {
                "status": "not_found",
                "route": f"{source} → {destination}",
                "verified": False,
                "confidence": 0.4,
                "count": 0,
                "all_news": [],
                "message": f"No news found for route {source} → {destination} in local DB."
            }

        # 4️⃣ Compute credibility & confidence
        credible_sources = list({n.get("source", "LocalDB") for n in results if n.get("source")})
        confidence = 0.9 if len(results) > 5 else 0.75

        return {
            "status": "success",
            "route": f"{source} → {destination}",
            "verified": True,
            "confidence": confidence,
            "count": len(results),
            "credible_sources": credible_sources,
            "all_news": results,
            "message": f"Fetched {len(results)} news items for route {source} → {destination} from local DB."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching route updates: {str(e)}")

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
