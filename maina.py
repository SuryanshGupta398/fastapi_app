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
import asyncio

# ---------------- Local imports ----------------
from configuration import collection, news_collection
from models import User, LoginUser
from gmail_service import send_email

# ---------------- FastAPI setup ----------------
app = FastAPI()
user_router = APIRouter(prefix="/users", tags=["Users"])
news_router = APIRouter(prefix="/news", tags=["News"])
report_router = APIRouter(prefix="/report", tags=["Report"])
verify_router = APIRouter(prefix="/verify", tags=["Verify"])

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ---------------- Load environment ----------------
NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY")
CRON_SECRET = os.getenv("CRON_SECRET")
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")

# ---------------- Load your trained models ----------------
print("🚀 Loading your trained ML models...")

try:
    # Load your fake news detection model
    fake_news_model = joblib.load("fake_news_model.pkl")
    tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
    print("✅ Fake news model loaded successfully!")
except Exception as e:
    print(f"⚠️ Error loading fake news model: {e}")
    fake_news_model = None
    tfidf_vectorizer = None

try:
    # Load your category classification model
    category_model = joblib.load("full_news_model.pkl")
    category_vectorizer = joblib.load("full_tfidf_vectorizer.pkl")
    category_encoder = joblib.load("label_encoder1.pkl")
    print("✅ Category model loaded successfully!")
except Exception as e:
    print(f"⚠️ Error loading category model: {e}")
    category_model = None
    category_vectorizer = None
    category_encoder = None

# ---------------- News Verification Service ----------------
# ---------------- Enhanced News Verification Service ----------------
class NewsVerificationService:
    def __init__(self):
        self.base_confidence = 0.5
    
    async def verify_news(self, news_text: str) -> dict:
        """Main verification function that searches all sources for historical news"""
        print(f"🔍 Verifying news (including historical): {news_text}")
        
        results = {
            "newsdata": {"found": False, "articles": [], "confidence_impact": 0, "time_range": "current"},
            "gnews": {"found": False, "articles": [], "confidence_impact": 0, "time_range": "current"},
            "database": {"found": False, "articles": [], "confidence_impact": 0, "time_range": "historical"},
            "ml_model": {"verdict": "UNCERTAIN", "confidence": 0.5, "confidence_impact": 0}
        }
        
        # Run all verification checks concurrently
        tasks = [
            self._check_newsdata_historical(news_text),
            self._check_gnews_historical(news_text),
            self._check_database_comprehensive(news_text),
            self._check_ml_model(news_text)
        ]
        
        newsdata_result, gnews_result, db_result, ml_result = await asyncio.gather(*tasks)
        
        # Update results
        results["newsdata"].update(newsdata_result)
        results["gnews"].update(gnews_result)
        results["database"].update(db_result)
        results["ml_model"].update(ml_result)
        
        # Calculate final confidence
        final_verdict, final_confidence = self._calculate_final_result(results)
        
        return {
            "search_results": results,
            "final_verdict": final_verdict,
            "final_confidence": final_confidence
        }
    
    async def _check_newsdata_historical(self, news_text: str) -> dict:
        """Check NewsData.io API with historical search capability"""
        try:
            if not NEWSDATA_API_KEY:
                return {"found": False, "articles": [], "confidence_impact": 0, "error": "API key missing", "time_range": "current"}
            
            # First try current news
            current_articles = await self._search_newsdata_timeframe(news_text, "current")
            
            if current_articles:
                return {
                    "found": True,
                    "articles": current_articles,
                    "confidence_impact": 0.3,
                    "count": len(current_articles),
                    "time_range": "current"
                }
            
            # If no current news found, try searching with broader time range
            # NewsData.io doesn't have direct historical search in free tier, but we can try without time filter
            url = "https://newsdata.io/api/1/news"
            params = {
                'apikey': NEWSDATA_API_KEY,
                'q': news_text[:500],
                'language': 'en',
                'size': 10  # Get more results to find historical matches
            }
            
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('results', [])
                
                if articles:
                    formatted_articles = []
                    for article in articles:
                        pub_date = article.get('pubDate', '')
                        time_range = self._classify_time_range(pub_date)
                        
                        formatted_articles.append({
                            'title': article.get('title', ''),
                            'description': article.get('description', ''),
                            'source': article.get('source_id', 'Unknown'),
                            'url': article.get('link', ''),
                            'published_at': pub_date,
                            'image_url': article.get('image_url', ''),
                            'time_period': time_range
                        })
                    
                    return {
                        "found": True,
                        "articles": formatted_articles[:5],  # Limit to 5
                        "confidence_impact": 0.25,  # Slightly less for potentially older news
                        "count": len(articles),
                        "time_range": "mixed"
                    }
            
            return {"found": False, "articles": [], "confidence_impact": 0, "count": 0, "time_range": "none"}
            
        except Exception as e:
            print(f"NewsData API error: {e}")
            return {"found": False, "articles": [], "confidence_impact": 0, "error": str(e), "time_range": "error"}
    
    async def _check_gnews_historical(self, news_text: str) -> dict:
        """Check GNews API with historical search"""
        try:
            if not GNEWS_API_KEY:
                return {"found": False, "articles": [], "confidence_impact": 0, "error": "API key missing", "time_range": "current"}
            
            # GNews search without time restriction to get historical results
            url = "https://gnews.io/api/v4/search"
            params = {
                'q': f'"{news_text[:100]}"',
                'token': GNEWS_API_KEY,
                'lang': 'en',
                'max': 10,  # Get more results
                'sortby': 'relevance'  # Sort by relevance rather than date
            }
            
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])
                
                if articles:
                    formatted_articles = []
                    for article in articles:
                        pub_date = article.get('publishedAt', '')
                        time_range = self._classify_time_range(pub_date)
                        
                        formatted_articles.append({
                            'title': article.get('title', ''),
                            'description': article.get('description', ''),
                            'source': article.get('source', {}).get('name', 'Unknown'),
                            'url': article.get('url', ''),
                            'published_at': pub_date,
                            'image_url': article.get('image', ''),
                            'time_period': time_range
                        })
                    
                    # Sort by date to show most recent first, but include historical
                    formatted_articles.sort(key=lambda x: x.get('published_at', ''), reverse=True)
                    
                    return {
                        "found": True,
                        "articles": formatted_articles[:5],
                        "confidence_impact": 0.25,
                        "count": len(articles),
                        "time_range": "mixed"
                    }
            
            return {"found": False, "articles": [], "confidence_impact": 0, "count": 0, "time_range": "none"}
            
        except Exception as e:
            print(f"GNews API error: {e}")
            return {"found": False, "articles": [], "confidence_impact": 0, "error": str(e), "time_range": "error"}
    
    async def _check_database_comprehensive(self, news_text: str) -> dict:
        """Check local MongoDB database for both current and historical news"""
        try:
            # Search in news collection without time restrictions
            query = {
                "$or": [
                    {"title": {"$regex": news_text, "$options": "i"}},
                    {"description": {"$regex": news_text, "$options": "i"}},
                    {"content": {"$regex": news_text, "$options": "i"}}
                ]
            }
            
            # Get both recent and historical articles
            articles = list(news_collection.find(query).sort("publishedAt", -1).limit(10))
            
            if articles:
                formatted_articles = []
                historical_count = 0
                current_count = 0
                
                for article in articles:
                    pub_date = article.get('publishedAt', '')
                    time_range = self._classify_time_range(pub_date)
                    
                    if time_range == "historical":
                        historical_count += 1
                    else:
                        current_count += 1
                    
                    formatted_articles.append({
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'source': article.get('source', 'Local Database'),
                        'url': article.get('url', ''),
                        'published_at': pub_date,
                        'category': article.get('category', 'Unknown'),
                        'image_url': article.get('image', ''),
                        'time_period': time_range
                    })
                
                # Determine time range for the result set
                if historical_count > 0 and current_count == 0:
                    time_range_label = "historical"
                    confidence_impact = 0.2  # Historical news still valuable for verification
                elif current_count > 0 and historical_count == 0:
                    time_range_label = "current"
                    confidence_impact = 0.25
                else:
                    time_range_label = "mixed"
                    confidence_impact = 0.25
                
                return {
                    "found": True,
                    "articles": formatted_articles,
                    "confidence_impact": confidence_impact,
                    "count": len(articles),
                    "time_range": time_range_label,
                    "historical_articles": historical_count,
                    "current_articles": current_count
                }
            
            return {"found": False, "articles": [], "confidence_impact": 0, "count": 0, "time_range": "none"}
            
        except Exception as e:
            print(f"Database search error: {e}")
            return {"found": False, "articles": [], "confidence_impact": 0, "error": str(e), "time_range": "error"}
    
    def _classify_time_range(self, date_string: str) -> str:
        """Classify article as current or historical based on publish date"""
        try:
            if not date_string:
                return "unknown"
            
            # Parse date (handle different formats)
            if 'T' in date_string:
                # ISO format: 2023-12-01T10:30:00Z
                pub_date = datetime.fromisoformat(date_string.replace('Z', '+00:00'))
            else:
                # Other formats, try common parsers
                for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%d/%m/%Y']:
                    try:
                        pub_date = datetime.strptime(date_string[:19], fmt)
                        break
                    except:
                        continue
                else:
                    return "unknown"
            
            # Calculate days difference
            now = datetime.utcnow()
            days_diff = (now - pub_date).days
            
            if days_diff <= 7:
                return "current"
            elif days_diff <= 30:
                return "recent"
            else:
                return "historical"
                
        except:
            return "unknown"
    
    async def _search_newsdata_timeframe(self, news_text: str, timeframe: str) -> list:
        """Helper method to search NewsData with specific timeframe"""
        try:
            url = "https://newsdata.io/api/1/news"
            params = {
                'apikey': NEWSDATA_API_KEY,
                'q': news_text[:500],
                'language': 'en',
                'size': 5
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('results', [])
                
                formatted_articles = []
                for article in articles:
                    pub_date = article.get('pubDate', '')
                    time_range = self._classify_time_range(pub_date)
                    
                    formatted_articles.append({
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'source': article.get('source_id', 'Unknown'),
                        'url': article.get('link', ''),
                        'published_at': pub_date,
                        'image_url': article.get('image_url', ''),
                        'time_period': time_range
                    })
                
                return formatted_articles
            return []
            
        except:
            return []
    
    async def _check_ml_model(self, news_text: str) -> dict:
        """Check with your trained ML model (unchanged)"""
        try:
            if fake_news_model is None or tfidf_vectorizer is None:
                return {
                    "verdict": "UNCERTAIN",
                    "confidence": 0.5,
                    "confidence_impact": 0,
                    "error": "ML model not available"
                }
            
            # Transform and predict
            text_vector = tfidf_vectorizer.transform([news_text])
            prediction = fake_news_model.predict(text_vector)[0]
            probability = fake_news_model.predict_proba(text_vector)[0]
            
            # Adjust based on your model's labeling
            if prediction == 1:  # Real
                verdict = "REAL"
                confidence = probability[1]
                impact = confidence * 0.25
            else:  # Fake
                verdict = "FAKE"
                confidence = probability[0]
                impact = -confidence * 0.25
            
            return {
                "verdict": verdict,
                "confidence": float(confidence),
                "confidence_impact": float(impact)
            }
            
        except Exception as e:
            print(f"ML model error: {e}")
            return {
                "verdict": "UNCERTAIN",
                "confidence": 0.5,
                "confidence_impact": 0,
                "error": str(e)
            }
    
    def _calculate_final_result(self, results: dict) -> tuple:
        """Calculate final verdict and confidence - ENHANCED for historical news"""
        total_impact = 0
        
        # Sum all confidence impacts
        total_impact += results["newsdata"]["confidence_impact"]
        total_impact += results["gnews"]["confidence_impact"] 
        total_impact += results["database"]["confidence_impact"]
        total_impact += results["ml_model"]["confidence_impact"]
        
        # Calculate final confidence
        final_confidence = self.base_confidence + total_impact
        final_confidence = max(0.0, min(1.0, final_confidence))
        
        # Enhanced verdict logic considering historical news
        ml_verdict = results["ml_model"]["verdict"]
        ml_confidence = results["ml_model"]["confidence"]
        
        # Count historical vs current findings
        historical_sources = 0
        current_sources = 0
        
        for source in ["newsdata", "gnews", "database"]:
            if results[source]["found"]:
                if results[source]["time_range"] in ["historical", "mixed"]:
                    historical_sources += 1
                elif results[source]["time_range"] == "current":
                    current_sources += 1
        
        # Verdict logic that values both historical and current verification
        if ml_confidence > 0.8:
            # High ML confidence overrides
            final_verdict = ml_verdict
        elif current_sources > 0:
            # Current verification is strongest
            if final_confidence >= 0.6:
                final_verdict = "REAL"
            elif final_confidence <= 0.4:
                final_verdict = "FAKE"
            else:
                final_verdict = "UNCERTAIN"
        elif historical_sources > 0:
            # Historical verification still valuable
            if final_confidence >= 0.65:
                final_verdict = "REAL"
            elif final_confidence <= 0.35:
                final_verdict = "FAKE"
            else:
                final_verdict = "UNCERTAIN"
        else:
            # No external verification, rely on ML
            if ml_confidence > 0.6:
                final_verdict = ml_verdict
            else:
                final_verdict = "UNCERTAIN"
        
        return final_verdict, round(final_confidence, 3)
# Initialize verification service
news_verifier = NewsVerificationService()

# ---------------- Request Models ----------------
class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    email: EmailStr
    otp: str
    new_password: str

class VerifyNewsRequest(BaseModel):
    news_text: str

# ---------------- Health Check ----------------
@app.get("/health")
def health_check():
    return {
        "status": "ok", 
        "time": datetime.utcnow().isoformat(),
        "models_loaded": {
            "fake_news_model": fake_news_model is not None,
            "category_model": category_model is not None
        }
    }

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

# ---------------- Forgot Password / Reset ----------------
@user_router.post("/forgot-password")
async def forgot_password(request: ForgotPasswordRequest, background_tasks: BackgroundTasks):
    user = collection.find_one({"email": request.email.lower()})
    if not user:
        raise HTTPException(status_code=404, detail="Email not registered")
    otp = str(random.randint(100000, 999999))
    otp_store[request.email.lower()] = {"otp": otp, "expires": datetime.utcnow() + timedelta(minutes=5)}
    background_tasks.add_task(send_otp_email, request.email, otp)
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

# ---------------- NEWS VERIFICATION ROUTES ----------------
@verify_router.post("/check-news")
async def check_news_comprehensive(news_text: str = Form(..., description="Type the news you want to verify")):
 @verify_router.post("/check-news")
async def check_news_comprehensive(news_text: str = Form(..., description="Type the news you want to verify")):
    """
    MAIN ENDPOINT: User types news and it searches ALL APIs and databases for current AND historical news
    """
    try:
        if not news_text.strip():
            raise HTTPException(status_code=400, detail="News text cannot be empty")
        
        print(f"🎯 User submitted news for verification: {news_text}")
        
        # Verify news using all sources (including historical)
        verification_result = await news_verifier.verify_news(news_text.strip())
        
        # Prepare enhanced response with time analysis
        response = {
            "status": "success",
            "user_input": news_text,
            "final_verdict": verification_result["final_verdict"],
            "confidence_score": verification_result["final_confidence"],
            "time_analysis": {
                "sources_with_current_news": 0,
                "sources_with_historical_news": 0,
                "overall_time_range": "unknown"
            },
            "sources_checked": {
                "newsdata_api": verification_result["search_results"]["newsdata"]["found"],
                "gnews_api": verification_result["search_results"]["gnews"]["found"],
                "local_database": verification_result["search_results"]["database"]["found"],
                "ml_model": True
            },
            "detailed_results": {
                "newsdata_api": {
                    "found": verification_result["search_results"]["newsdata"]["found"],
                    "articles_count": verification_result["search_results"]["newsdata"]["count"],
                    "time_range": verification_result["search_results"]["newsdata"]["time_range"],
                    "sample_articles": verification_result["search_results"]["newsdata"]["articles"][:2]
                },
                "gnews_api": {
                    "found": verification_result["search_results"]["gnews"]["found"],
                    "articles_count": verification_result["search_results"]["gnews"]["count"],
                    "time_range": verification_result["search_results"]["gnews"]["time_range"],
                    "sample_articles": verification_result["search_results"]["gnews"]["articles"][:2]
                },
                "local_database": {
                    "found": verification_result["search_results"]["database"]["found"],
                    "articles_count": verification_result["search_results"]["database"]["count"],
                    "time_range": verification_result["search_results"]["database"]["time_range"],
                    "sample_articles": verification_result["search_results"]["database"]["articles"][:2]
                },
                "ml_analysis": {
                    "verdict": verification_result["search_results"]["ml_model"]["verdict"],
                    "confidence": verification_result["search_results"]["ml_model"]["confidence"]
                }
            },
            "verdict_explanation": get_verdict_explanation(verification_result),
            "search_metadata": {
                "searched_historical": True,
                "total_sources_checked": 4,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        return response
        
    except Exception as e:
        print(f"Error in check_news_comprehensive: {e}")
        raise HTTPException(status_code=500, detail=f"News verification failed: {str(e)}")


def get_verdict_explanation(verification_result: dict) -> str:
    """Generate human-readable explanation considering historical news"""
    verdict = verification_result["final_verdict"]
    confidence = verification_result["final_confidence"]
    results = verification_result["search_results"]
    
    # Count sources and time ranges
    sources_found = []
    historical_sources = 0
    current_sources = 0
    
    for source_name in ["newsdata", "gnews", "database"]:
        source_data = results[source_name]
        if source_data["found"]:
            sources_found.append(source_name.capitalize())
            if source_data["time_range"] == "current":
                current_sources += 1
            elif source_data["time_range"] in ["historical", "mixed"]:
                historical_sources += 1
    
    ml_verdict = results["ml_model"]["verdict"]
    ml_confidence = results["ml_model"]["confidence"]
    
    if verdict == "REAL":
        if current_sources > 0:
            return f"✅ VERIFIED: This recent news appears authentic and was confirmed by {len(sources_found)} sources including {', '.join(sources_found)}."
        elif historical_sources > 0:
            return f"✅ HISTORICALLY VERIFIED: This news was reported by {len(sources_found)} sources in the past and appears authentic. Our AI analysis confirms credibility."
        else:
            return f"✅ LIKELY REAL: Our AI analysis indicates this news is credible with {ml_confidence:.0%} confidence."
    
    elif verdict == "FAKE":
        if ml_confidence > 0.8:
            return f"❌ LIKELY FALSE: Our AI detection strongly indicates misinformation patterns. Multiple verification attempts found no credible sources."
        else:
            return f"❌ SUSPICIOUS: This content shows characteristics of unreliable news. No credible verification found across {len(sources_found)} searched sources."
    
    else:  # UNCERTAIN
        if historical_sources > 0:
            return f"⚠️ HISTORICAL REFERENCE FOUND: This news was reported in the past but current verification is limited. Check updated sources for confirmation."
        elif len(sources_found) > 0:
            return f"⚠️ NEEDS VERIFICATION: Found in {len(sources_found)} sources but our analysis is inconclusive. The news might be outdated or from limited sources."
        else:
            return f"⚠️ UNCERTAIN: We cannot verify this news. It may be very recent, from limited sources, or historical. Check multiple reliable sources."# ---------------- Batch Verification ----------------

@verify_router.post("/check-multiple-news")
async def check_multiple_news(
    news_items: list[str] = Form(..., description="List of news items to verify")
):
    """Verify multiple news items at once"""
    try:
        results = []
        for news_text in news_items:
            if news_text.strip():
                result = await check_news_comprehensive(news_text)
                results.append({
                    "news_text": news_text,
                    "verdict": result["final_verdict"],
                    "confidence": result["confidence_score"],
                    "sources_found": sum([
                        1 for source in result["sources_checked"].values() 
                        if source is True
                    ])
                })
        
        return {
            "status": "success",
            "total_checked": len(results),
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch verification failed: {str(e)}")

# ---------------- JSON API Endpoint ----------------
@verify_router.post("/check-news-json")
async def check_news_json(request: VerifyNewsRequest):
    """JSON version of news verification"""
    return await check_news_comprehensive(request.news_text)

# ---------------- Register Routers ----------------
app.include_router(user_router)
app.include_router(news_router)
app.include_router(report_router)
app.include_router(verify_router)

print("✅ Fake News Detector Backend Started Successfully!")
print("📡 Available Verification Endpoints:")
print("   POST /verify/check-news          - Main endpoint (Form data)")
print("   POST /verify/check-news-json     - JSON API endpoint")
print("   POST /verify/check-multiple-news - Batch verification")
