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
class NewsVerificationService:
    def __init__(self):
        self.base_confidence = 0.5
    
    async def verify_news(self, news_text: str) -> dict:
        """Main verification function that searches all sources"""
        print(f"🔍 Verifying news: {news_text}")
        
        results = {
            "newsdata": {"found": False, "articles": [], "confidence_impact": 0},
            "gnews": {"found": False, "articles": [], "confidence_impact": 0},
            "database": {"found": False, "articles": [], "confidence_impact": 0},
            "ml_model": {"verdict": "UNCERTAIN", "confidence": 0.5, "confidence_impact": 0}
        }
        
        # Run all verification checks concurrently
        tasks = [
            self._check_newsdata(news_text),
            self._check_gnews(news_text),
            self._check_database(news_text),
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
    
    async def _check_newsdata(self, news_text: str) -> dict:
        """Check with NewsData.io API"""
        try:
            if not NEWSDATA_API_KEY:
                return {"found": False, "articles": [], "confidence_impact": 0, "error": "API key missing"}
            
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
                
                if articles:
                    formatted_articles = []
                    for article in articles:
                        formatted_articles.append({
                            'title': article.get('title', ''),
                            'description': article.get('description', ''),
                            'source': article.get('source_id', 'Unknown'),
                            'url': article.get('link', ''),
                            'published_at': article.get('pubDate', ''),
                            'image_url': article.get('image_url', '')
                        })
                    
                    return {
                        "found": True,
                        "articles": formatted_articles,
                        "confidence_impact": 0.3,
                        "count": len(articles)
                    }
            
            return {"found": False, "articles": [], "confidence_impact": -0.2, "count": 0}
            
        except Exception as e:
            print(f"NewsData API error: {e}")
            return {"found": False, "articles": [], "confidence_impact": 0, "error": str(e)}
    
    async def _check_gnews(self, news_text: str) -> dict:
        """Check with GNews API"""
        try:
            if not GNEWS_API_KEY:
                return {"found": False, "articles": [], "confidence_impact": 0, "error": "API key missing"}
            
            url = "https://gnews.io/api/v4/search"
            params = {
                'q': f'"{news_text[:100]}"',
                'token': GNEWS_API_KEY,
                'lang': 'en',
                'max': 5
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])
                
                if articles:
                    formatted_articles = []
                    for article in articles:
                        formatted_articles.append({
                            'title': article.get('title', ''),
                            'description': article.get('description', ''),
                            'source': article.get('source', {}).get('name', 'Unknown'),
                            'url': article.get('url', ''),
                            'published_at': article.get('publishedAt', ''),
                            'image_url': article.get('image', '')
                        })
                    
                    return {
                        "found": True,
                        "articles": formatted_articles,
                        "confidence_impact": 0.25,
                        "count": len(articles)
                    }
            
            return {"found": False, "articles": [], "confidence_impact": -0.15, "count": 0}
            
        except Exception as e:
            print(f"GNews API error: {e}")
            return {"found": False, "articles": [], "confidence_impact": 0, "error": str(e)}
    
    async def _check_database(self, news_text: str) -> dict:
        """Check with local MongoDB database"""
        try:
            # Search in news collection
            query = {
                "$or": [
                    {"title": {"$regex": news_text, "$options": "i"}},
                    {"description": {"$regex": news_text, "$options": "i"}}
                ]
            }
            
            articles = list(news_collection.find(query).limit(5))
            
            if articles:
                formatted_articles = []
                for article in articles:
                    formatted_articles.append({
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'source': article.get('source', 'Local Database'),
                        'url': article.get('url', ''),
                        'published_at': article.get('publishedAt', ''),
                        'category': article.get('category', 'Unknown'),
                        'image_url': article.get('image', '')
                    })
                
                return {
                    "found": True,
                    "articles": formatted_articles,
                    "confidence_impact": 0.2,
                    "count": len(articles)
                }
            
            return {"found": False, "articles": [], "confidence_impact": -0.1, "count": 0}
            
        except Exception as e:
            print(f"Database search error: {e}")
            return {"found": False, "articles": [], "confidence_impact": 0, "error": str(e)}
    
    async def _check_ml_model(self, news_text: str) -> dict:
        """Check with your trained ML model"""
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
        """Calculate final verdict and confidence"""
        total_impact = 0
        
        # Sum all confidence impacts
        total_impact += results["newsdata"]["confidence_impact"]
        total_impact += results["gnews"]["confidence_impact"]
        total_impact += results["database"]["confidence_impact"]
        total_impact += results["ml_model"]["confidence_impact"]
        
        # Calculate final confidence
        final_confidence = self.base_confidence + total_impact
        final_confidence = max(0.0, min(1.0, final_confidence))
        
        # Determine verdict
        if final_confidence >= 0.7:
            verdict = "REAL"
        elif final_confidence <= 0.3:
            verdict = "FAKE"
        else:
            verdict = "UNCERTAIN"
        
        return verdict, round(final_confidence, 3)

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
    """
    MAIN ENDPOINT: User types news and it searches ALL APIs and databases
    """
    try:
        if not news_text.strip():
            raise HTTPException(status_code=400, detail="News text cannot be empty")
        
        print(f"🎯 User submitted news for verification: {news_text}")
        
        # Verify news using all sources
        verification_result = await news_verifier.verify_news(news_text.strip())
        
        # Prepare response
        response = {
            "status": "success",
            "user_input": news_text,
            "final_verdict": verification_result["final_verdict"],
            "confidence_score": verification_result["final_confidence"],
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
                    "sample_articles": verification_result["search_results"]["newsdata"]["articles"][:2]
                },
                "gnews_api": {
                    "found": verification_result["search_results"]["gnews"]["found"],
                    "articles_count": verification_result["search_results"]["gnews"]["count"],
                    "sample_articles": verification_result["search_results"]["gnews"]["articles"][:2]
                },
                "local_database": {
                    "found": verification_result["search_results"]["database"]["found"],
                    "articles_count": verification_result["search_results"]["database"]["count"],
                    "sample_articles": verification_result["search_results"]["database"]["articles"][:2]
                },
                "ml_analysis": {
                    "verdict": verification_result["search_results"]["ml_model"]["verdict"],
                    "confidence": verification_result["search_results"]["ml_model"]["confidence"]
                }
            },
            "confidence_breakdown": {
                "base_score": 0.5,
                "newsdata_impact": verification_result["search_results"]["newsdata"]["confidence_impact"],
                "gnews_impact": verification_result["search_results"]["gnews"]["confidence_impact"],
                "database_impact": verification_result["search_results"]["database"]["confidence_impact"],
                "ml_model_impact": verification_result["search_results"]["ml_model"]["confidence_impact"],
                "total_impact": sum([
                    verification_result["search_results"]["newsdata"]["confidence_impact"],
                    verification_result["search_results"]["gnews"]["confidence_impact"],
                    verification_result["search_results"]["database"]["confidence_impact"],
                    verification_result["search_results"]["ml_model"]["confidence_impact"]
                ])
            },
            "verdict_explanation": get_verdict_explanation(verification_result),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return response
        
    except Exception as e:
        print(f"Error in check_news_comprehensive: {e}")
        raise HTTPException(status_code=500, detail=f"News verification failed: {str(e)}")

def get_verdict_explanation(verification_result: dict) -> str:
    """Generate human-readable explanation"""
    verdict = verification_result["final_verdict"]
    confidence = verification_result["final_confidence"]
    results = verification_result["search_results"]
    
    sources_found = []
    if results["newsdata"]["found"]:
        sources_found.append("NewsData")
    if results["gnews"]["found"]:
        sources_found.append("GNews")
    if results["database"]["found"]:
        sources_found.append("Local Database")
    
    if verdict == "REAL":
        if confidence >= 0.8:
            return f"✅ HIGHLY CREDIBLE: This news appears authentic and was verified across {len(sources_found)} sources including {', '.join(sources_found)}. The ML model also confirms its authenticity with high confidence."
        else:
            return f"✅ LIKELY REAL: This news appears credible based on verification from {len(sources_found)} sources. The ML analysis supports this conclusion."
    
    elif verdict == "FAKE":
        if confidence <= 0.2:
            return f"❌ HIGHLY SUSPICIOUS: This news was not found in credible sources and shows patterns consistent with misinformation. ML analysis strongly indicates fake content."
        else:
            return f"❌ LIKELY FAKE: Limited verification from credible sources. The news shows characteristics of unreliable content according to our ML model."
    
    else:  # UNCERTAIN
        return f"⚠️ UNCERTAIN: We need more information to verify this news. It was found in {len(sources_found)} sources. Please check multiple reliable sources before sharing."

# ---------------- Batch Verification ----------------
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
