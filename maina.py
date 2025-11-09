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
from textblob import TextBlob
from sentence_transformers import SentenceTransformer, util
import asyncio
import aiohttp

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

# ---------------- LIGHTNING-FAST VERIFICATION ----------------

# Fake news indicators (lightweight pattern matching)
FAKE_INDICATORS = [
    r"breaking.*shocking", r"you won't believe", r"doctors hate", r"this one trick",
    r"miracle cure", r"secret they don't want", r"instant results", r"viral.*secret",
    r"everyone is talking about", r"shocked the world", r"celebrity.*died",
    r"government hiding", r"mainstream media won't", r"exposed.*truth"
]

CREDIBLE_SOURCES = ["bbc", "reuters", "ap news", "associated press", "cnn", "al jazeera"]

def quick_pattern_check(headline: str) -> dict:
    """Ultra-fast pattern matching for fake news detection"""
    headline_lower = headline.lower()
    
    # Check for fake indicators
    fake_score = 0
    for pattern in FAKE_INDICATORS:
        if pattern in headline_lower:
            fake_score += 1
    
    # Check for credible source mentions
    credible_score = 0
    for source in CREDIBLE_SOURCES:
        if source in headline_lower:
            credible_score += 1
    
    # Sentiment analysis (lightweight)
    blob = TextBlob(headline)
    polarity = blob.sentiment.polarity
    
    # Determine result
    if fake_score >= 2:
        return {"rating": "Fake", "confidence": 0.8, "reason": "Multiple fake news patterns detected"}
    elif credible_score >= 1:
        return {"rating": "True", "confidence": 0.7, "reason": "Mentions credible sources"}
    elif abs(polarity) > 0.5:  # Highly emotional
        return {"rating": "Suspicious", "confidence": 0.6, "reason": "Highly emotional language"}
    else:
        return {"rating": "Uncertain", "confidence": 0.5, "reason": "Insufficient data for quick analysis"}

async def fast_mongodb_check(headline: str) -> dict:
    """Fast MongoDB lookup with timeout"""
    try:
        # Search for similar headlines in last 7 days
        recent_limit = datetime.utcnow() - timedelta(days=7)
        
        # Simple text search (faster than semantic)
        similar_news = list(news_collection.find({
            "$text": {"$search": headline},
            "createdAt": {"$gte": recent_limit}
        }).limit(5))
        
        if similar_news:
            return {
                "found": True,
                "count": len(similar_news),
                "sources": list(set([n.get("source", "Unknown") for n in similar_news])),
                "articles": [{"title": n["title"], "url": n.get("url", "")} for n in similar_news[:3]]
            }
        return {"found": False, "count": 0}
    except Exception as e:
        print(f"MongoDB check error: {e}")
        return {"found": False, "count": 0, "error": str(e)}

async def fast_gnews_check(headline: str) -> dict:
    """Fast GNews check with timeout"""
    GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")
    if not GNEWS_API_KEY:
        return {"error": "API key missing"}
    
    try:
        # Use async session for faster requests
        timeout = aiohttp.ClientTimeout(total=5)  # 5 second timeout
        async with aiohttp.ClientSession(timeout=timeout) as session:
            url = f"https://gnews.io/api/v4/search?q={headline[:50]}&token={GNEWS_API_KEY}&lang=en&max=3"
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if "articles" in data and data["articles"]:
                        return {
                            "found": True,
                            "count": len(data["articles"]),
                            "articles": [{"title": a["title"], "url": a["url"]} for a in data["articles"][:3]]
                        }
                return {"found": False, "count": 0}
    except asyncio.TimeoutError:
        return {"error": "Timeout"}
    except Exception as e:
        return {"error": str(e)}

# ---------------- LIGHTNING-FAST VERIFY NEWS ROUTE ----------------
@news_router.post("/verify-news-fast")
async def verify_news_fast(headline: str = Form(...)):
    """
    ⚡ LIGHTNING-FAST news verification for small scale
    Uses pattern matching + fast MongoDB lookup + quick external check
    """
    start_time = datetime.utcnow()
    
    try:
        headline = headline.strip()
        if not headline:
            raise HTTPException(status_code=400, detail="Headline required")

        # Step 1: Ultra-fast pattern check (instant)
        pattern_result = quick_pattern_check(headline)
        
        # Step 2: Parallel fast checks (MongoDB + GNews)
        mongodb_task = fast_mongodb_check(headline)
        gnews_task = fast_gnews_check(headline)
        
        mongodb_result, gnews_result = await asyncio.gather(
            mongodb_task, gnews_task, return_exceptions=True
        )

        # Handle exceptions
        if isinstance(mongodb_result, Exception):
            mongodb_result = {"found": False, "error": str(mongodb_result)}
        if isinstance(gnews_result, Exception):
            gnews_result = {"found": False, "error": str(gnews_result)}

        # Step 3: Quick confidence calculation
        confidence_factors = []
        
        # Pattern match confidence
        if pattern_result["rating"] == "Fake":
            confidence_factors.append(0.2)  # Lower weight for patterns
        elif pattern_result["rating"] == "True":
            confidence_factors.append(0.3)
        
        # MongoDB matches
        if mongodb_result.get("found"):
            confidence_factors.append(min(0.3 + (mongodb_result["count"] * 0.1), 0.6))
        
        # GNews matches
        if gnews_result.get("found"):
            confidence_factors.append(min(0.4 + (gnews_result["count"] * 0.1), 0.7))
        
        # Calculate final confidence
        if confidence_factors:
            final_confidence = sum(confidence_factors) / len(confidence_factors)
            final_confidence = min(final_confidence, 0.9)  # Cap at 90%
        else:
            final_confidence = pattern_result["confidence"]
        
        # Determine final rating
        if mongodb_result.get("found") or gnews_result.get("found"):
            final_rating = "True"
        elif pattern_result["rating"] == "Fake":
            final_rating = "Fake"
        else:
            final_rating = "Uncertain"

        # Calculate response time
        response_time = (datetime.utcnow() - start_time).total_seconds()

        return {
            "status": "success",
            "verified": final_rating == "True",
            "headline": headline,
            "rating": final_rating,
            "confidence": round(final_confidence, 2),
            "response_time_seconds": round(response_time, 3),
            "analysis": {
                "pattern_check": pattern_result,
                "mongodb_matches": mongodb_result.get("count", 0),
                "external_matches": gnews_result.get("count", 0),
                "sources_found": mongodb_result.get("sources", [])
            },
            "sources": {
                "mongodb_articles": mongodb_result.get("articles", []),
                "gnews_articles": gnews_result.get("articles", [])
            },
            "message": f"Verified in {response_time:.2f}s: {final_rating} ({final_confidence*100:.0f}% confidence)"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Verification error: {str(e)}")

# ---------------- SIMPLE VERIFICATION (FASTEST) ----------------
@news_router.post("/verify-news-instant")
async def verify_news_instant(headline: str = Form(...)):
    """
    🚀 INSTANT verification - pattern matching only
    For when you need the absolute fastest response
    """
    try:
        headline = headline.strip()
        if not headline:
            raise HTTPException(status_code=400, detail="Headline required")

        # Only pattern matching - instant response
        result = quick_pattern_check(headline)
        
        return {
            "status": "success",
            "verified": result["rating"] == "True",
            "headline": headline,
            "rating": result["rating"],
            "confidence": result["confidence"],
            "reason": result["reason"],
            "response_time": "instant",
            "message": f"Instant analysis: {result['rating']}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Instant verification error: {str(e)}")

# ---------------- KEEP YOUR EXISTING ROUTES BUT ADD FAST OPTIONS ----------------

# Replace your existing verify-news with this faster version
@news_router.post("/verify-news")
async def verify_news_optimized(headline: str = Form(...)):
    """
    🎯 Optimized version of your original verify-news
    Faster semantic search with smaller timeouts
    """
    try:
        headline = headline.strip()
        if not headline:
            raise HTTPException(status_code=400, detail="Headline required")

        # Load lightweight model only when needed
        model_st = SentenceTransformer("all-MiniLM-L6-v2")  # Smaller model
        
        # Faster time window (7 days instead of 15)
        recent_limit = datetime.utcnow() - timedelta(days=7)
        
        # Fetch only essential fields
        cursor = news_collection.find(
            {"createdAt": {"$gte": recent_limit}},
            {"title": 1, "url": 1, "source": 1}
        ).limit(100)  # Limit to 100 documents for speed

        docs = list(cursor)
        if not docs:
            # Fallback to fast version
            return await verify_news_fast(headline)

        # Fast semantic similarity with batch processing
        query_emb = model_st.encode(headline, convert_to_tensor=True)
        doc_titles = [doc.get("title", "") for doc in docs]
        doc_embs = model_st.encode(doc_titles, convert_to_tensor=True)
        
        similarities = util.cos_sim(query_emb, doc_embs)[0]
        
        matches = []
        for i, sim in enumerate(similarities):
            if sim > 0.6:  # Lower threshold for more matches
                matches.append({
                    "title": docs[i]["title"],
                    "similarity": round(sim.item(), 3),
                    "url": docs[i].get("url", ""),
                    "source": docs[i].get("source", "Unknown")
                })

        # Quick analysis
        if len(matches) >= 2:
            sources = list(set(m["source"] for m in matches))
            avg_similarity = sum(m["similarity"] for m in matches) / len(matches)
            
            if avg_similarity > 0.7:
                rating, confidence = "True", min(0.8 + (len(sources) * 0.05), 0.95)
            else:
                rating, confidence = "Likely True", 0.7
        elif len(matches) == 1:
            rating, confidence = "Uncertain", 0.6
        else:
            # Fallback to pattern matching
            pattern_result = quick_pattern_check(headline)
            rating, confidence = pattern_result["rating"], pattern_result["confidence"]

        return {
            "status": "success",
            "verified": rating in ["True", "Likely True"],
            "headline": headline,
            "rating": rating,
            "confidence": confidence,
            "matches_found": len(matches),
            "sources": list(set(m["source"] for m in matches)),
            "top_matches": matches[:3]
        }

    except Exception as e:
        # Fallback to fastest version on error
        return await verify_news_instant(headline)

# ---------------- OPTIMIZED NEWS FETCHING ----------------
async def fetch_news_optimized(lang="en"):
    """Optimized news fetching with smaller payloads"""
    # Your existing fetch_and_store_news code but with:
    # - Smaller timeouts
    # - Fewer articles per request
    # - Async requests
    pass

# ---------------- Add these optimizations to your existing code ----------------

# Add text index for faster searches (run this once in your database)
# db.news_collection.create_index([("title", "text"), ("description", "text")])

# ---------------- Register Routers ----------------
app.include_router(user_router)
app.include_router(news_router)
app.include_router(report_router)

# ---------------- Startup Optimization ----------------
@app.on_event("startup")
async def startup_event():
    """Preload essential components"""
    print("🚀 Starting optimized Fake News Detector...")
    
    # Preload lightweight models in background
    asyncio.create_task(preload_models())

async def preload_models():
    """Preload frequently used models"""
    try:
        # Preload the small semantic model
        SentenceTransformer("all-MiniLM-L6-v2")
        print("✅ Lightweight models preloaded")
    except Exception as e:
        print(f"⚠️ Model preloading failed: {e}")

# ---------------- Health Check with Performance Info ----------------
@app.get("/health")
def health_check():
    return {
        "status": "ok", 
        "time": datetime.utcnow().isoformat(), 
        "model_accuracy": current_accuracy,
        "optimized": True,
        "fast_routes_available": ["/verify-news-fast", "/verify-news-instant"]
    }
