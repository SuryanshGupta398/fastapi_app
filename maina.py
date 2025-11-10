import os
import requests
import re
from datetime import datetime, timedelta
from fastapi import FastAPI, APIRouter, HTTPException, Form

# ---------------- Local imports ----------------
from configuration import collection, news_collection
from models import User, LoginUser
from gmail_service import send_email

# ---------------- FastAPI setup ----------------
app = FastAPI()
user_router = APIRouter(prefix="/users", tags=["Users"])
news_router = APIRouter(prefix="/news", tags=["News"])
report_router = APIRouter(prefix="/report", tags=["Report"])

# ---------------- Load environment ----------------
NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY")
CRON_SECRET = os.getenv("CRON_SECRET")

print("✅ Lightweight Fake News Detector Ready!")

# ---------------- PATTERN MATCHING (No ML) ----------------

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

def simple_text_similarity(text1, text2):
    """Simple word overlap similarity"""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    common = words1.intersection(words2)
    return len(common) / max(len(words1), len(words2))

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
    """Fast MongoDB check using simple text matching"""
    try:
        # Search last 3 days only for speed
        recent_limit = datetime.utcnow() - timedelta(days=3)
        
        # Get all recent news
        recent_news = list(news_collection.find({
            "createdAt": {"$gte": recent_limit}
        }).limit(50))
        
        if not recent_news:
            return {"found": False, "count": 0}
        
        # Simple similarity matching
        matches = []
        for news in recent_news:
            news_title = news.get("title", "").lower()
            if not news_title:
                continue
                
            # Simple word overlap check
            similarity = simple_text_similarity(headline, news_title)
            if similarity > 0.3:  # 30% word overlap
                matches.append({
                    "title": news.get("title", ""),
                    "similarity": round(similarity, 2),
                    "url": news.get("url", ""),
                    "source": news.get("source", "Unknown")
                })
        
        # Sort by similarity and take top 3
        matches.sort(key=lambda x: x["similarity"], reverse=True)
        top_matches = matches[:3]
        
        return {
            "found": len(top_matches) > 0,
            "count": len(top_matches),
            "matches": top_matches
        }
        
    except Exception as e:
        print(f"MongoDB check error: {e}")
        return {"found": False, "count": 0}

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

# ---------------- MAIN VERIFICATION ROUTE ----------------
@news_router.post("/verify-news")
async def verify_news_comprehensive(headline: str = Form(...)):
    """
    🔍 COMPREHENSIVE verification - pattern + DB + external news
    Response time: 3-5 seconds
    """
    start_time = datetime.utcnow()
    
    try:
        headline = headline.strip()
        if not headline:
            raise HTTPException(status_code=400, detail="Headline required")

        # Step 1: Instant pattern check
        pattern_result = ultra_fast_pattern_check(headline)
        
        # Step 2: MongoDB check
        mongodb_result = fast_mongodb_check(headline)
        
        # Step 3: External news check
        gnews_result = fast_gnews_check(headline)
        
        # Step 4: Calculate comprehensive confidence
        confidence_factors = [pattern_result["confidence"]]
        
        # MongoDB influence
        if mongodb_result.get("found"):
            db_boost = 0.15 + (mongodb_result["count"] * 0.05)
            confidence_factors.append(min(pattern_result["confidence"] + db_boost, 0.9))
        
        # External news influence
        if gnews_result.get("found"):
            external_boost = 0.2 + (gnews_result["count"] * 0.08)
            if gnews_result.get("trusted_sources", 0) > 0:
                external_boost += 0.1  # Extra boost for trusted domains
            confidence_factors.append(min(pattern_result["confidence"] + external_boost, 0.95))
        
        # Calculate final score
        final_confidence = sum(confidence_factors) / len(confidence_factors)
        final_confidence = min(final_confidence, 0.95)
        
        # Determine final rating
        if gnews_result.get("found") and mongodb_result.get("found"):
            final_rating = "Verified True"
        elif gnews_result.get("found") or mongodb_result.get("found"):
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
                "external_news": gnews_result.get("count", 0),
                "trusted_sources": gnews_result.get("trusted_sources", 0)
            },
            "evidence": {
                "pattern_result": pattern_result,
                "database_matches": mongodb_result.get("matches", []),
                "external_articles": gnews_result.get("articles", [])
            },
            "message": f"Analysis completed in {response_time:.1f}s: {final_rating} ({final_confidence*100}% confidence)"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Verification error: {str(e)}")

# ---------------- SIMPLIFIED USER ROUTES ----------------
@user_router.post("/register")
async def register_user(new_user: User):
    email = new_user.email.strip().lower()
    if collection.find_one({"username": new_user.username}):
        raise HTTPException(status_code=400, detail="Username already taken")
    if collection.find_one({"email": email}):
        raise HTTPException(status_code=400, detail="Email already registered")

    user_dict = new_user.dict()
    user_dict["email"] = email
    user_dict["password"] = new_user.password  # Remove hashing for simplicity
    user_dict["created_at"] = datetime.utcnow()
    resp = collection.insert_one(user_dict)

    return {"status": "success", "id": str(resp.inserted_id), "message": "User registered successfully"}

@user_router.post("/signin")
async def signin_user(login_user: User):
    email = login_user.email.strip().lower()
    user_in_db = collection.find_one({"email": email})
    if not user_in_db or login_user.password != user_in_db["password"]:
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

# ---------------- Register Routers ----------------
app.include_router(user_router)
app.include_router(news_router)
app.include_router(report_router)

# ---------------- Health Check ----------------
@app.get("/health")
def health_check():
    return {
        "status": "ok", 
        "time": datetime.utcnow().isoformat(),
        "message": "Lightweight Fake News Detector API"
    }

@app.get("/")
def root():
    return {"message": "Fake News Detector API", "version": "1.0"}
