from fastapi import FastAPI, APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from configuration import collection
from models import User, LoginUser
from datetime import datetime, timedelta
from passlib.context import CryptContext
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig, MessageType
import os, random

app = FastAPI()
user_router = APIRouter(prefix="/users", tags=["Users"])

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ------------------ Email Config ------------------
conf = ConnectionConfig(
    MAIL_USERNAME=os.getenv("MAIL_USERNAME"),
    MAIL_PASSWORD=os.getenv("MAIL_PASSWORD"),  # Gmail App Password
    MAIL_FROM=os.getenv("MAIL_FROM"),
    MAIL_PORT=587,
    MAIL_SERVER="smtp.gmail.com",
    MAIL_STARTTLS=True,
    MAIL_SSL_TLS=False,
    USE_CREDENTIALS=True
)

fm = FastMail(conf)

# ------------------ SIGN IN ------------------
@user_router.post("/signin")
async def signin_user(login_user: LoginUser):
    user_in_db = collection.find_one({"email": login_user.email})
    if not user_in_db:
        return JSONResponse(status_code=401, content={"message": "Invalid email or password"})

    if not pwd_context.verify(login_user.password, user_in_db["password"]):
        return JSONResponse(status_code=401, content={"message": "Invalid email or password"})

    return JSONResponse(
        status_code=200,
        content={
            "status_code": 200,
            "message": "Login successful",
            "full_name": user_in_db.get("full_name", ""),
            "username": user_in_db.get("username", ""),
            "email": user_in_db["email"]
        }
    )

# ------------------ REGISTER ------------------
@user_router.post("/register")
async def register_user(new_user: User):
    try:
        if collection.find_one({"username": new_user.username}):
            return JSONResponse(status_code=400, content={"message": "Username already taken"})
        if collection.find_one({"email": new_user.email}):
            return JSONResponse(status_code=400, content={"message": "Email already registered"})

        user_dict = new_user.model_dump()
        user_dict["created_at"] = datetime.utcnow()
        user_dict["password"] = pwd_context.hash(new_user.password)
        resp = collection.insert_one(user_dict)

        # 📧 Welcome Email
        message = MessageSchema(
            subject="Welcome to My App 🎉",
            recipients=[new_user.email],
            body=f"""
                <h2>Hello {new_user.full_name},</h2>
                Thank you for signing up with Fake News Detector ! We’re excited to have you join our mission of making the internet a safer and more reliable place.

With Fake News Detector , you can:
✅ Instantly check if a news article is genuine or misleading
✅ Stay updated with verified news sources
✅ Report suspicious content and help others avoid misinformation
✅ Be part of a growing community that values truth and transparency

Whether you’re a student, researcher, or just someone who wants to know the facts, our app gives you the tools you need to separate fact from fiction.

What’s next?

Open the app and explore your dashboard

Try scanning your first news article

Share the app with friends and family so they can stay safe too

Together, we can make a real difference against the spread of fake news.
Welcome once again, and thank you for trusting us.

Best regards,
The Multiverse Team
            """,
            subtype=MessageType.html
        )
        await fm.send_message(message)

        return JSONResponse(
            status_code=200,
            content={"status": "success", "id": str(resp.inserted_id),
                     "message": "User registered successfully and email sent"}
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Error: {str(e)}"})

# ------------------ FORGOT PASSWORD ------------------
@user_router.post("/forgot-password")
async def forgot_password(email: str = Query(...)):
    user = collection.find_one({"email": email})
    if not user:
        raise HTTPException(status_code=404, detail="Email not registered")

    otp = str(random.randint(100000, 999999))
    expiry = datetime.utcnow() + timedelta(minutes=5)

    collection.update_one(
        {"email": email},
        {"$set": {"reset_otp": otp, "reset_expiry": expiry}}
    )

    message = MessageSchema(
        subject="Password Reset OTP",
        recipients=[email],
        body=f"""
            <h2>Password Reset Request</h2>
            <p>Your OTP is: <b>{otp}</b></p>
            <p>This OTP is valid for 5 minutes.</p>
        """,
        subtype=MessageType.html
    )
    await fm.send_message(message)

    return {"status": "success", "message": "OTP sent to email"}

# ------------------ RESET PASSWORD ------------------
@user_router.post("/reset-password")
async def reset_password(email: str, otp: str, new_password: str):
    user = collection.find_one({"email": email})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if "reset_otp" not in user or "reset_expiry" not in user:
        raise HTTPException(status_code=400, detail="OTP not generated")

    if datetime.utcnow() > user["reset_expiry"]:
        raise HTTPException(status_code=400, detail="OTP expired")

    if otp != user["reset_otp"]:
        raise HTTPException(status_code=400, detail="Invalid OTP")

    hashed_password = pwd_context.hash(new_password)
    collection.update_one(
        {"email": email},
        {"$set": {"password": hashed_password}, "$unset": {"reset_otp": "", "reset_expiry": ""}}
    )

    return {"status": "success", "message": "Password reset successfully"}

# ------------------ DELETE ------------------
@user_router.delete("/delete/{email}")
async def delete_user(email: str):
    result = collection.delete_one({"email": email})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
    return {"status": "success", "message": "Account deleted"}

app.include_router(user_router)
