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
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from textblob import TextBlob
from typing import Dict, List, Optional
from pymongo import MongoClient

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

# ---------------- FastAPI setup ----------------
app = FastAPI()
user_router = APIRouter(prefix="/users", tags=["Users"])
news_router = APIRouter(prefix="/news", tags=["News"])
report_router = APIRouter(prefix="/report", tags=["Report"])
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

# ---------------- Semantic Analyzer ----------------
class SemanticAnalyzer:
    def __init__(self):
        # Negation words and patterns
        self.negation_words = {
            'not', 'no', 'never', 'nothing', 'none', 'nobody', 'nowhere', 'neither',
            'cannot', "can't", "won't", "wouldn't", "couldn't", "shouldn't", "isn't",
            "aren't", "wasn't", "weren't", "don't", "doesn't", "didn't", "haven't",
            "hasn't", "hadn't", 'without', 'lack', 'missing', 'absent'
        }
        
        # Opposite word pairs
        self.opposite_pairs = {
            'good': 'bad', 'bad': 'good', 'positive': 'negative', 'negative': 'positive',
            'true': 'false', 'false': 'true', 'real': 'fake', 'fake': 'real',
            'yes': 'no', 'no': 'yes', 'accept': 'reject', 'reject': 'accept',
            'support': 'oppose', 'oppose': 'support', 'win': 'lose', 'lose': 'win',
            'increase': 'decrease', 'decrease': 'increase', 'up': 'down', 'down': 'up',
            'success': 'failure', 'failure': 'success', 'right': 'wrong', 'wrong': 'right',
            'confirm': 'deny', 'deny': 'confirm', 'prove': 'disprove', 'disprove': 'prove',
            'found': 'lost', 'lost': 'found', 'approve': 'reject', 'reject': 'approve'
        }
        
        # Contradiction indicators
        self.contradiction_indicators = {
            'but', 'however', 'although', 'though', 'while', 'whereas', 'nevertheless',
            'on the other hand', 'in contrast', 'conversely', 'despite', 'in spite of'
        }

    def analyze_semantics(self, text: str) -> Dict:
        """Comprehensive semantic analysis of news text"""
        text_lower = text.lower()
        
        analysis = {
            'has_negation': False,
            'negation_context': [],
            'opposite_words_detected': [],
            'contradictions_found': [],
            'sentiment_score': 0,
            'subjectivity_score': 0,
            'word_count': len(text.split()),
            'sentence_count': len(sent_tokenize(text)),
            'semantic_risk_score': 0,
            'certainty_level': 'medium',
            'linguistic_analysis': {}
        }
        
        # 1. Detect negations
        negation_analysis = self._detect_negations(text)
        analysis.update(negation_analysis)
        
        # 2. Detect opposite words
        opposite_analysis = self._detect_opposites(text)
        analysis.update(opposite_analysis)
        
        # 3. Detect contradictions
        contradiction_analysis = self._detect_contradictions(text)
        analysis.update(contradiction_analysis)
        
        # 4. Sentiment analysis
        sentiment_analysis = self._analyze_sentiment(text)
        analysis.update(sentiment_analysis)
        
        # 5. Linguistic features
        linguistic_analysis = self._analyze_linguistic_features(text)
        analysis['linguistic_analysis'] = linguistic_analysis
        
        # 6. Calculate risk score and certainty
        analysis['semantic_risk_score'] = self._calculate_risk_score(analysis)
        analysis['certainty_level'] = self._determine_certainty(analysis)
        
        return analysis
    
    def _detect_negations(self, text: str) -> Dict:
        """Detect negation patterns"""
        text_lower = text.lower()
        words = text_lower.split()
        
        negations_found = []
        for i, word in enumerate(words):
            if word in self.negation_words:
                # Get context (next 2-3 words)
                context_end = min(i + 4, len(words))
                context = ' '.join(words[i:context_end])
                negations_found.append({
                    'negation_word': word,
                    'context': context,
                    'position': i
                })
        
        return {
            'has_negation': len(negations_found) > 0,
            'negation_context': negations_found
        }
    
    def _detect_opposites(self, text: str) -> Dict:
        """Detect opposite words"""
        text_lower = text.lower()
        words = set(text_lower.split())
        
        opposites_found = []
        for word in words:
            if word in self.opposite_pairs:
                opposites_found.append({
                    'word': word,
                    'opposite': self.opposite_pairs[word],
                    'impact': 'high' if word in ['not', 'no', 'never'] else 'medium'
                })
        
        return {'opposite_words_detected': opposites_found}
    
    def _detect_contradictions(self, text: str) -> Dict:
        """Detect contradiction patterns"""
        text_lower = text.lower()
        
        contradictions_found = []
        for indicator in self.contradiction_indicators:
            if indicator in text_lower:
                contradictions_found.append({
                    'indicator': indicator,
                    'context': self._get_contradiction_context(text_lower, indicator)
                })
        
        # Detect "X but Y" pattern specifically
        if ' but ' in text_lower:
            but_context = self._extract_but_pattern(text_lower)
            if but_context:
                contradictions_found.append({
                    'indicator': 'but',
                    'context': but_context
                })
        
        return {
            'contradictions_found': contradictions_found,
            'has_contradictions': len(contradictions_found) > 0
        }
    
    def _analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment using TextBlob"""
        try:
            blob = TextBlob(text)
            return {
                'sentiment_score': blob.sentiment.polarity,
                'subjectivity_score': blob.sentiment.subjectivity,
                'sentiment_label': self._classify_sentiment(blob.sentiment.polarity)
            }
        except:
            return {
                'sentiment_score': 0,
                'subjectivity_score': 0,
                'sentiment_label': 'neutral'
            }
    
    def _analyze_linguistic_features(self, text: str) -> Dict:
        """Analyze linguistic features"""
        text_lower = text.lower()
        
        # Count specific patterns
        exclamation_count = text.count('!')
        question_count = text.count('?')
        capital_ratio = sum(1 for c in text if c.isupper()) / max(1, len(text))
        
        # Check for clickbait patterns
        clickbait_phrases = [
            'you won\'t believe', 'this will shock you', 'what happened next',
            'doctors hate this', 'the truth about', 'they don\'t want you to know'
        ]
        
        clickbait_count = sum(1 for phrase in clickbait_phrases if phrase in text_lower)
        
        # Check for sensational language
        sensational_words = [
            'shocking', 'amazing', 'unbelievable', 'incredible', 'miracle',
            'secret', 'breaking', 'urgent', 'alert', 'warning'
        ]
        
        sensational_count = sum(1 for word in sensational_words if word in text_lower)
        
        return {
            'exclamation_count': exclamation_count,
            'question_count': question_count,
            'capital_ratio': round(capital_ratio, 3),
            'clickbait_indicators': clickbait_count,
            'sensational_language': sensational_count,
            'has_clickbait': clickbait_count > 0,
            'has_sensational_language': sensational_count > 0
        }
    
    def _get_contradiction_context(self, text: str, indicator: str) -> str:
        """Get context around contradiction indicator"""
        try:
            index = text.find(indicator)
            start = max(0, index - 50)
            end = min(len(text), index + 50)
            return text[start:end].strip()
        except:
            return indicator
    
    def _extract_but_pattern(self, text: str) -> str:
        """Extract 'but' contradiction pattern"""
        try:
            # Simple pattern extraction
            parts = text.split(' but ', 1)
            if len(parts) == 2:
                return f"{parts[0].strip()} BUT {parts[1].strip()}"
            return ""
        except:
            return ""
    
    def _classify_sentiment(self, score: float) -> str:
        """Classify sentiment score"""
        if score > 0.1:
            return 'positive'
        elif score < -0.1:
            return 'negative'
        else:
            return 'neutral'
    
    def _calculate_risk_score(self, analysis: Dict) -> float:
        """Calculate semantic risk score (0-1)"""
        risk_score = 0.0
        
        # Negation increases risk
        if analysis['has_negation']:
            risk_score += 0.3
        
        # Multiple negations increase risk more
        if len(analysis['negation_context']) > 1:
            risk_score += 0.2
        
        # Opposite words and contradictions increase risk
        risk_score += min(0.3, len(analysis['opposite_words_detected']) * 0.1)
        risk_score += min(0.2, len(analysis['contradictions_found']) * 0.1)
        
        # High sentiment polarity indicates potential bias
        if abs(analysis['sentiment_score']) > 0.5:
            risk_score += 0.2
        
        # Clickbait and sensational language increase risk
        linguistic = analysis['linguistic_analysis']
        if linguistic['has_clickbait']:
            risk_score += 0.15
        if linguistic['has_sensational_language']:
            risk_score += 0.15
        
        # Excessive punctuation
        if linguistic['exclamation_count'] > 2:
            risk_score += 0.1
        
        return min(1.0, risk_score)
    
    def _determine_certainty(self, analysis: Dict) -> str:
        """Determine certainty level based on semantic analysis"""
        risk_score = analysis['semantic_risk_score']
        
        if risk_score > 0.6:
            return 'low'
        elif risk_score > 0.3:
            return 'medium'
        else:
            return 'high'

# Initialize semantic analyzer
semantic_analyzer = SemanticAnalyzer()

# ---------------- News Verification Service ----------------
class NewsVerificationService:
    def __init__(self):
        self.base_confidence = 0.5
    
    async def verify_news(self, news_text: str) -> dict:
        """Main verification function with semantic analysis"""
        print(f"🔍 Verifying news with semantic analysis: {news_text}")
        
        # First, perform semantic analysis
        semantic_analysis = semantic_analyzer.analyze_semantics(news_text)
        
        results = {
            "newsdata": {"found": False, "articles": [], "confidence_impact": 0, "time_range": "current"},
            "gnews": {"found": False, "articles": [], "confidence_impact": 0, "time_range": "current"},
            "database": {"found": False, "articles": [], "confidence_impact": 0, "time_range": "historical"},
            "ml_model": {"verdict": "UNCERTAIN", "confidence": 0.5, "confidence_impact": 0},
            "semantic_analysis": semantic_analysis
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
        
        # Calculate final confidence with semantic consideration
        final_verdict, final_confidence = self._calculate_final_result(results)
        
        return {
            "search_results": results,
            "final_verdict": final_verdict,
            "final_confidence": final_confidence,
            "semantic_analysis": semantic_analysis
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
            url = "https://newsdata.io/api/1/news"
            params = {
                'apikey': NEWSDATA_API_KEY,
                'q': news_text[:500],
                'language': 'en',
                'size': 10
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
                        "articles": formatted_articles[:5],
                        "confidence_impact": 0.25,
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
                'max': 10,
                'sortby': 'relevance'
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
                    confidence_impact = 0.2
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
        """Calculate final verdict considering semantic meaning"""
        total_impact = 0
        
        # Sum all confidence impacts
        total_impact += results["newsdata"]["confidence_impact"]
        total_impact += results["gnews"]["confidence_impact"] 
        total_impact += results["database"]["confidence_impact"]
        total_impact += results["ml_model"]["confidence_impact"]
        
        # Adjust for semantic risk
        semantic_risk = results["semantic_analysis"]["semantic_risk_score"]
        semantic_adjustment = -semantic_risk * 0.3
        total_impact += semantic_adjustment
        
        # Calculate final confidence
        final_confidence = self.base_confidence + total_impact
        final_confidence = max(0.0, min(1.0, final_confidence))
        
        # Enhanced verdict logic considering semantics
        ml_verdict = results["ml_model"]["verdict"]
        ml_confidence = results["ml_model"]["confidence"]
        semantic_certainty = results["semantic_analysis"]["certainty_level"]
        
        # High semantic risk can override other signals
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
    # Implement your email sending logic here
    print(f"Would send welcome email to: {email}")

async def send_otp_email(email: str, otp: str):
    subject = "Password Reset OTP"
    body = f"<h2>Password Reset</h2><p>Your OTP is: <b>{otp}</b></p><p>Valid for 5 minutes.</p>"
    # Implement your email sending logic here
    print(f"Would send OTP email to: {email}")

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
def get_verdict_explanation(verification_result: dict) -> str:
    """Generate human-readable explanation considering semantic meaning"""
    verdict = verification_result["final_verdict"]
    confidence = verification_result["final_confidence"]
    results = verification_result["search_results"]
    semantic = verification_result["semantic_analysis"]
    
    # Count sources
    sources_found = []
    for source_name in ["newsdata", "gnews", "database"]:
        if results[source_name]["found"]:
            sources_found.append(source_name.capitalize())
    
    # Semantic analysis factors
    has_negation = semantic['has_negation']
    has_contradictions = semantic['has_contradictions']
    certainty = semantic['certainty_level']
    semantic_risk = semantic['semantic_risk_score']
    
    if verdict == "REAL":
        if certainty == 'high':
            return f"✅ CLEARLY VERIFIED: This news is semantically clear and was confirmed by {len(sources_found)} sources."
        elif has_negation:
            return f"✅ VERIFIED (with nuance): The news contains negations but was confirmed by {len(sources_found)} sources. Pay attention to the specific wording."
        else:
            return f"✅ LIKELY REAL: Verified by {len(sources_found)} sources with reasonable semantic clarity."
    
    elif verdict == "FAKE":
        if has_contradictions:
            return f"❌ CONTRADICTORY: This news contains internal contradictions and shows patterns of misinformation."
        elif has_negation and len(sources_found) == 0:
            return f"❌ LIKELY FALSE: The negative phrasing combined with lack of verification suggests misinformation."
        elif semantic_risk > 0.6:
            return f"❌ HIGH SEMANTIC RISK: The language shows strong indicators of unreliable content."
        else:
            return f"❌ SUSPICIOUS: Shows characteristics of unreliable content with low verification."
    
    else:  # UNCERTAIN
        if has_contradictions:
            return f"⚠️ SEMANTICALLY UNCLEAR: This news contains contradictions that make verification difficult."
        elif has_negation:
            return f"⚠️ NEEDS CAREFUL READING: Contains negations that could change meaning. Verify the exact phrasing."
        elif certainty == 'low':
            return f"⚠️ AMBIGUOUS WORDING: The language used makes this difficult to verify conclusively."
        else:
            return f"⚠️ UNCERTAIN: Limited verification available. Check multiple sources."

@verify_router.post("/check-news")
async def check_news_comprehensive(news_text: str = Form(..., description="Type the news you want to verify")):
    """
    MAIN ENDPOINT: User types news and it searches ALL APIs with semantic understanding
    """
    try:
        if not news_text.strip():
            raise HTTPException(status_code=400, detail="News text cannot be empty")
        
        print(f"🎯 User submitted news for semantic verification: {news_text}")
        
        # Verify news using all sources with semantic analysis
        verification_result = await news_verifier.verify_news(news_text.strip())
        
        # Prepare enhanced response with semantic analysis
        response = {
            "status": "success",
            "user_input": news_text,
            "final_verdict": verification_result["final_verdict"],
            "confidence_score": verification_result["final_confidence"],
            "semantic_analysis": {
                "has_negation": verification_result["semantic_analysis"]["has_negation"],
                "negation_context": verification_result["semantic_analysis"]["negation_context"],
                "opposite_words_detected": verification_result["semantic_analysis"]["opposite_words_detected"],
                "contradictions_found": verification_result["semantic_analysis"]["contradictions_found"],
                "semantic_risk_score": round(verification_result["semantic_analysis"]["semantic_risk_score"], 3),
                "certainty_level": verification_result["semantic_analysis"]["certainty_level"],
                "sentiment_analysis": {
                    "score": round(verification_result["semantic_analysis"]["sentiment_score"], 3),
                    "label": verification_result["semantic_analysis"]["sentiment_label"],
                    "subjectivity": round(verification_result["semantic_analysis"]["subjectivity_score"], 3)
                },
                "linguistic_analysis": verification_result["semantic_analysis"]["linguistic_analysis"]
            },
            "sources_checked": {
                "newsdata_api": verification_result["search_results"]["newsdata"]["found"],
                "gnews_api": verification_result["search_results"]["gnews"]["found"],
                "local_database": verification_result["search_results"]["database"]["found"],
                "ml_model": True,
                "semantic_analysis": True
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
                "total_sources_checked": 5,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        return response
        
    except Exception as e:
        print(f"Error in check_news_comprehensive: {e}")
        raise HTTPException(status_code=500, detail=f"News verification failed: {str(e)}")

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

# ---------------- Semantic Analysis Only Endpoint ----------------
@verify_router.post("/analyze-semantics")
async def analyze_semantics_only(news_text: str = Form(..., description="Analyze the semantics of news text")):
    """Endpoint to get only semantic analysis"""
    try:
        if not news_text.strip():
            raise HTTPException(status_code=400, detail="News text cannot be empty")
        
        analysis = semantic_analyzer.analyze_semantics(news_text.strip())
        
        return {
            "status": "success",
            "news_text": news_text,
            "semantic_analysis": analysis
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Semantic analysis failed: {str(e)}")

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
print("   POST /verify/analyze-semantics   - Semantic analysis only")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
