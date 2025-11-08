import os
import requests
from datetime import datetime, timedelta
import random
import joblib
import numpy as np
from fastapi import FastAPI, APIRouter, HTTPException, BackgroundTasks, Form
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import re
import nltk
from nltk.tokenize import sent_tokenize
from textblob import TextBlob
from typing import Dict, List
from pymongo import MongoClient

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
except:
    pass

# ---------------- FastAPI setup ----------------
app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
user_router = APIRouter(prefix="/users", tags=["Users"])
verify_router = APIRouter(prefix="/verify", tags=["Verify"])

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ---------------- Database Setup ----------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/fake_news_detector")
client = MongoClient(MONGO_URI)
db = client.fake_news_detector
collection = db.users
news_collection = db.news

# ---------------- Load environment ----------------
NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY")
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")

# ---------------- Load your trained models ----------------
print("🚀 Loading your trained ML models...")

try:
    fake_news_model = joblib.load("fake_news_model.pkl")
    tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
    ML_MODELS_LOADED = True
    print("✅ Fake news model loaded successfully!")
except Exception as e:
    print(f"⚠️ Error loading fake news model: {e}")
    fake_news_model = None
    tfidf_vectorizer = None
    ML_MODELS_LOADED = False

# ---------------- Semantic Analyzer ----------------
class SemanticAnalyzer:
    def __init__(self):
        self.negation_words = {
            'not', 'no', 'never', 'nothing', 'none', 'nobody', 'without',
            'cannot', "can't", "won't", "wouldn't", "couldn't", "shouldn't", "isn't",
            "aren't", "wasn't", "weren't", "don't", "doesn't", "didn't"
        }
        
        self.opposite_pairs = {
            'good': 'bad', 'bad': 'good', 'positive': 'negative', 'negative': 'positive',
            'true': 'false', 'false': 'true', 'real': 'fake', 'fake': 'real',
            'yes': 'no', 'no': 'yes', 'accept': 'reject', 'reject': 'accept',
            'support': 'oppose', 'oppose': 'support', 'win': 'lose', 'lose': 'win'
        }

    def analyze_semantics(self, text: str) -> Dict:
        """Lightweight semantic analysis"""
        text_lower = text.lower()
        
        analysis = {
            'has_negation': False,
            'negation_context': [],
            'opposite_words_detected': [],
            'sentiment_score': 0,
            'word_count': len(text.split()),
            'semantic_risk_score': 0,
        }
        
        # Detect negations
        words = text_lower.split()
        negations_found = []
        for i, word in enumerate(words):
            if word in self.negation_words:
                context_end = min(i + 4, len(words))
                context = ' '.join(words[i:context_end])
                negations_found.append({
                    'negation_word': word,
                    'context': context,
                })
        
        analysis['has_negation'] = len(negations_found) > 0
        analysis['negation_context'] = negations_found
        
        # Detect opposite words
        words_set = set(text_lower.split())
        opposites_found = []
        for word in words_set:
            if word in self.opposite_pairs:
                opposites_found.append({
                    'word': word,
                    'opposite': self.opposite_pairs[word],
                })
        
        analysis['opposite_words_detected'] = opposites_found
        
        # Simple sentiment analysis
        try:
            blob = TextBlob(text)
            analysis['sentiment_score'] = blob.sentiment.polarity
            analysis['sentiment_label'] = 'positive' if blob.sentiment.polarity > 0.1 else 'negative' if blob.sentiment.polarity < -0.1 else 'neutral'
        except:
            analysis['sentiment_score'] = 0
            analysis['sentiment_label'] = 'neutral'
        
        # Calculate risk score
        risk_score = 0.0
        if analysis['has_negation']:
            risk_score += 0.3
        risk_score += len(analysis['opposite_words_detected']) * 0.1
        if abs(analysis['sentiment_score']) > 0.5:
            risk_score += 0.2
        
        analysis['semantic_risk_score'] = min(1.0, risk_score)
        analysis['certainty_level'] = 'low' if risk_score > 0.6 else 'medium' if risk_score > 0.3 else 'high'
        
        return analysis

# Initialize semantic analyzer
semantic_analyzer = SemanticAnalyzer()

# ---------------- News Verification Service ----------------
class NewsVerificationService:
    def __init__(self):
        self.base_confidence = 0.5
    
    async def verify_news(self, news_text: str) -> dict:
        """Main verification function"""
        print(f"🔍 Verifying news: {news_text}")
        
        # Perform semantic analysis
        semantic_analysis = semantic_analyzer.analyze_semantics(news_text)
        
        results = {
            "newsdata": {"found": False, "articles": [], "confidence_impact": 0},
            "gnews": {"found": False, "articles": [], "confidence_impact": 0},
            "database": {"found": False, "articles": [], "confidence_impact": 0},
            "ml_model": {"verdict": "UNCERTAIN", "confidence": 0.5, "confidence_impact": 0},
            "semantic_analysis": semantic_analysis
        }
        
        # Run verification checks
        tasks = [
            self._check_newsdata(news_text),
            self._check_gnews(news_text),
            self._check_database(news_text),
            self._check_ml_model(news_text)
        ]
        
        newsdata_result, gnews_result, db_result, ml_result = await asyncio.gather(*tasks)
        
        results["newsdata"].update(newsdata_result)
        results["gnews"].update(gnews_result)
        results["database"].update(db_result)
        results["ml_model"].update(ml_result)
        
        # Calculate final confidence
        final_verdict, final_confidence = self._calculate_final_result(results)
        
        return {
            "search_results": results,
            "final_verdict": final_verdict,
            "final_confidence": final_confidence,
            "semantic_analysis": semantic_analysis
        }
    
    async def _check_newsdata(self, news_text: str) -> dict:
        """Check NewsData.io API"""
        try:
            if not NEWSDATA_API_KEY:
                return {"found": False, "articles": [], "confidence_impact": 0}
            
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
                    for article in articles[:3]:  # Limit to 3
                        formatted_articles.append({
                            'title': article.get('title', ''),
                            'description': article.get('description', ''),
                            'source': article.get('source_id', 'Unknown'),
                            'url': article.get('link', ''),
                        })
                    
                    return {
                        "found": True,
                        "articles": formatted_articles,
                        "confidence_impact": 0.3,
                        "count": len(articles)
                    }
            
            return {"found": False, "articles": [], "confidence_impact": 0, "count": 0}
            
        except Exception as e:
            print(f"NewsData API error: {e}")
            return {"found": False, "articles": [], "confidence_impact": 0}
    
    async def _check_gnews(self, news_text: str) -> dict:
        """Check GNews API"""
        try:
            if not GNEWS_API_KEY:
                return {"found": False, "articles": [], "confidence_impact": 0}
            
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
                    for article in articles[:3]:
                        formatted_articles.append({
                            'title': article.get('title', ''),
                            'description': article.get('description', ''),
                            'source': article.get('source', {}).get('name', 'Unknown'),
                            'url': article.get('url', ''),
                        })
                    
                    return {
                        "found": True,
                        "articles": formatted_articles,
                        "confidence_impact": 0.25,
                        "count": len(articles)
                    }
            
            return {"found": False, "articles": [], "confidence_impact": 0, "count": 0}
            
        except Exception as e:
            print(f"GNews API error: {e}")
            return {"found": False, "articles": [], "confidence_impact": 0}
    
    async def _check_database(self, news_text: str) -> dict:
        """Check local MongoDB database"""
        try:
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
                    })
                
                return {
                    "found": True,
                    "articles": formatted_articles,
                    "confidence_impact": 0.2,
                    "count": len(articles)
                }
            
            return {"found": False, "articles": [], "confidence_impact": 0, "count": 0}
            
        except Exception as e:
            print(f"Database search error: {e}")
            return {"found": False, "articles": [], "confidence_impact": 0}
    
    async def _check_ml_model(self, news_text: str) -> dict:
        """Check with ML model"""
        try:
            if fake_news_model is None or tfidf_vectorizer is None:
                return {
                    "verdict": "UNCERTAIN",
                    "confidence": 0.5,
                    "confidence_impact": 0
                }
            
            text_vector = tfidf_vectorizer.transform([news_text])
            prediction = fake_news_model.predict(text_vector)[0]
            probability = fake_news_model.predict_proba(text_vector)[0]
            
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
                "confidence_impact": 0
            }
    
    def _calculate_final_result(self, results: dict) -> tuple:
        """Calculate final verdict"""
        total_impact = 0
        
        total_impact += results["newsdata"]["confidence_impact"]
        total_impact += results["gnews"]["confidence_impact"] 
        total_impact += results["database"]["confidence_impact"]
        total_impact += results["ml_model"]["confidence_impact"]
        
        # Adjust for semantic risk
        semantic_risk = results["semantic_analysis"]["semantic_risk_score"]
        total_impact -= semantic_risk * 0.3
        
        final_confidence = self.base_confidence + total_impact
        final_confidence = max(0.0, min(1.0, final_confidence))
        
        ml_verdict = results["ml_model"]["verdict"]
        ml_confidence = results["ml_model"]["confidence"]
        
        if semantic_risk > 0.7:
            final_verdict = "UNCERTAIN"
        elif ml_confidence > 0.8:
            final_verdict = ml_verdict
        elif final_confidence >= 0.6:
            final_verdict = "REAL"
        elif final_confidence <= 0.4:
            final_verdict = "FAKE"
        else:
            final_verdict = "UNCERTAIN"
        
        return final_verdict, round(final_confidence, 3)

# Initialize verification service
news_verifier = NewsVerificationService()

# ---------------- Request Models ----------------
class User(BaseModel):
    username: str
    email: EmailStr
    password: str
    full_name: str

class LoginUser(BaseModel):
    email: EmailStr
    password: str

class VerifyNewsRequest(BaseModel):
    news_text: str

# ---------------- Health Check ----------------
@app.get("/")
async def root():
    return {
        "message": "Fake News Detector API", 
        "status": "running",
        "models_loaded": ML_MODELS_LOADED,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# ---------------- Email Helpers ----------------
async def send_welcome_email(email: str, full_name: str):
    print(f"📧 Welcome email would be sent to: {email}")

async def send_otp_email(email: str, otp: str):
    print(f"📧 OTP email would be sent to: {email}")

otp_store = {}

# ---------------- User Routes ----------------
@user_router.post("/register")
async def register_user(user: User, background_tasks: BackgroundTasks):
    email = user.email.strip().lower()
    if collection.find_one({"username": user.username}):
        raise HTTPException(status_code=400, detail="Username already taken")
    if collection.find_one({"email": email}):
        raise HTTPException(status_code=400, detail="Email already registered")

    user_dict = user.dict()
    user_dict["email"] = email
    user_dict["password"] = pwd_context.hash(user.password)
    user_dict["created_at"] = datetime.utcnow()
    resp = collection.insert_one(user_dict)

    background_tasks.add_task(send_welcome_email, email, user.full_name)
    return {"status": "success", "id": str(resp.inserted_id), "message": "User registered successfully"}

@user_router.post("/login")
async def login_user(login_user: LoginUser):
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

# ---------------- NEWS VERIFICATION ROUTES ----------------
def get_verdict_explanation(verification_result: dict) -> str:
    """Generate explanation"""
    verdict = verification_result["final_verdict"]
    results = verification_result["search_results"]
    semantic = verification_result["semantic_analysis"]
    
    sources_found = []
    for source_name in ["newsdata", "gnews", "database"]:
        if results[source_name]["found"]:
            sources_found.append(source_name.capitalize())
    
    has_negation = semantic['has_negation']
    
    if verdict == "REAL":
        if has_negation:
            return f"✅ VERIFIED (with nuance): Contains negations but confirmed by {len(sources_found)} sources."
        else:
            return f"✅ VERIFIED: Confirmed by {len(sources_found)} sources."
    
    elif verdict == "FAKE":
        if has_negation:
            return f"❌ LIKELY FALSE: Negative phrasing suggests misinformation."
        else:
            return f"❌ SUSPICIOUS: Shows characteristics of unreliable content."
    
    else:
        if has_negation:
            return f"⚠️ NEEDS VERIFICATION: Contains negations that affect meaning."
        else:
            return f"⚠️ UNCERTAIN: Limited verification available."

@verify_router.post("/check-news")
async def check_news_comprehensive(news_text: str = Form(..., description="Type the news you want to verify")):
    """Main verification endpoint"""
    try:
        if not news_text.strip():
            raise HTTPException(status_code=400, detail="News text cannot be empty")
        
        print(f"🎯 User submitted news: {news_text}")
        
        verification_result = await news_verifier.verify_news(news_text.strip())
        
        response = {
            "status": "success",
            "user_input": news_text,
            "final_verdict": verification_result["final_verdict"],
            "confidence_score": verification_result["final_confidence"],
            "semantic_analysis": {
                "has_negation": verification_result["semantic_analysis"]["has_negation"],
                "semantic_risk_score": round(verification_result["semantic_analysis"]["semantic_risk_score"], 3),
                "sentiment": verification_result["semantic_analysis"]["sentiment_label"]
            },
            "sources_checked": {
                "newsdata_api": verification_result["search_results"]["newsdata"]["found"],
                "gnews_api": verification_result["search_results"]["gnews"]["found"],
                "local_database": verification_result["search_results"]["database"]["found"],
                "ml_model": True,
                "semantic_analysis": True
            },
            "verdict_explanation": get_verdict_explanation(verification_result),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return response
        
    except Exception as e:
        print(f"Error in check_news_comprehensive: {e}")
        raise HTTPException(status_code=500, detail=f"News verification failed: {str(e)}")

@verify_router.post("/check-multiple-news")
async def check_multiple_news(news_items: list[str] = Form(..., description="List of news items to verify")):
    """Verify multiple news items"""
    try:
        results = []
        for news_text in news_items:
            if news_text.strip():
                result = await check_news_comprehensive(news_text)
                results.append({
                    "news_text": news_text,
                    "verdict": result["final_verdict"],
                    "confidence": result["confidence_score"],
                })
        
        return {
            "status": "success",
            "total_checked": len(results),
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch verification failed: {str(e)}")

@verify_router.post("/check-news-json")
async def check_news_json(request: VerifyNewsRequest):
    """JSON version of news verification"""
    return await check_news_comprehensive(request.news_text)

# ---------------- Register Routers ----------------
app.include_router(user_router)
app.include_router(verify_router)

print("✅ Fake News Detector Backend Started Successfully!")
print("📡 Available Endpoints:")
print("   POST /verify/check-news          - Main endpoint")
print("   POST /verify/check-news-json     - JSON API endpoint")
print("   POST /verify/check-multiple-news - Batch verification")
print("   POST /users/register             - User registration")
print("   POST /users/login                - User login")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
