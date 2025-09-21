from fastapi import FastAPI, APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from configuration import collection
from models import User, LoginUser
from datetime import datetime, timedelta
from passlib.context import CryptContext
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig, MessageType
from jose import JWTError, jwt
import os, random

app = FastAPI()
user_router = APIRouter(prefix="/users", tags=["Users"])

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ------------------ JWT CONFIG ------------------
SECRET_KEY = os.getenv("SECRET_KEY", "supersecretkey")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="users/signin")


def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


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


# ------------------ REQUEST MODELS ------------------
class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    email: EmailStr
    otp: str
    new_password: str


# ------------------ SIGN IN ------------------
@user_router.post("/signin")
async def signin_user(form_data: OAuth2PasswordRequestForm = Depends()):
    email = form_data.username.strip().lower()
    password = form_data.password

    user_in_db = collection.find_one({"email": email})
    if not user_in_db or not pwd_context.verify(password, user_in_db["password"]):
        return JSONResponse(status_code=401, content={"message": "Invalid email or password"})

    access_token = create_access_token(data={"sub": email})

    return JSONResponse(
        status_code=200,
        content={
            "status": "success",
            "message": "Login successful",
            "access_token": access_token,
            "token_type": "bearer",
            "full_name": user_in_db.get("full_name", ""),
            "username": user_in_db.get("username", ""),
            "email": user_in_db["email"]
        }
    )


# ------------------ REGISTER ------------------
@user_router.post("/register")
async def register_user(new_user: User):
    try:
        new_user.email = new_user.email.strip().lower()

        if collection.find_one({"username": new_user.username}):
            return JSONResponse(status_code=400, content={"message": "Username already taken"})
        if collection.find_one({"email": new_user.email}):
            return JSONResponse(status_code=400, content={"message": "Email already registered"})

        user_dict = new_user.model_dump()
        user_dict["created_at"] = datetime.utcnow()
        user_dict["password"] = str(pwd_context.hash(new_user.password))
        resp = collection.insert_one(user_dict)

        # 📧 Welcome Email
        message = MessageSchema(
            subject="Welcome to Fake News Detector 🎉",
            recipients=[new_user.email],
            body=f"""
                <h2>Hello {new_user.full_name},</h2>
                <p>Thank you for signing up with <b>Fake News Detector</b>! 
                We’re excited to have you join our mission of making the internet safer.</p>
                <p>With our app you can:</p>
                <ul>
                    <li>✅ Instantly check if a news article is genuine</li>
                    <li>✅ Stay updated with verified news sources</li>
                    <li>✅ Report suspicious content</li>
                </ul>
                <p>Welcome once again, and thank you for trusting us.</p>
                <br>
                <p>Best regards,<br>The Multiverse Team</p>
            """,
            subtype=MessageType.html
        )
        await fm.send_message(message)

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "id": str(resp.inserted_id),
                "message": "User registered successfully and email sent"
            }
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Error: {str(e)}"})


# ------------------ FORGOT PASSWORD ------------------
@user_router.post("/forgot-password")
async def forgot_password(request: ForgotPasswordRequest):
    email = request.email.strip().lower()
    user = collection.find_one({"email": email})
    if not user:
        raise HTTPException(status_code=404, detail="Email not registered")

    otp = str(random.randint(100000, 999999))
    expiry = datetime.utcnow() + timedelta(minutes=5)

    # Overwrite old OTP (no duplicates)
    collection.update_one(
        {"email": email},
        {"$set": {"reset_otp": otp, "reset_expiry": expiry}},
        upsert=False
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

    return JSONResponse(status_code=200, content={"status": "success", "message": "OTP sent to email"})


# ------------------ RESET PASSWORD ------------------
@user_router.post("/reset-password")
async def reset_password(request: ResetPasswordRequest):
    email = request.email.strip().lower()
    otp = request.otp
    new_password = request.new_password

    user = collection.find_one({"email": email})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    otp_in_db = user.get("reset_otp")
    expiry_in_db = user.get("reset_expiry")

    if not otp_in_db or not expiry_in_db:
        raise HTTPException(status_code=400, detail="OTP not generated or already used")

    if datetime.utcnow() > expiry_in_db:
        raise HTTPException(status_code=400, detail="OTP expired")

    if otp != otp_in_db:
        raise HTTPException(status_code=400, detail="Invalid OTP")

    hashed_password = pwd_context.hash(new_password)
    collection.update_one(
        {"email": email},
        {"$set": {"password": hashed_password}, "$unset": {"reset_otp": 1, "reset_expiry": 1}}
    )

    return JSONResponse(status_code=200, content={"status": "success", "message": "Password reset successfully"})


# ------------------ DELETE ------------------
@user_router.delete("/delete/{email}")
async def delete_user(email: str):
    email = email.strip().lower()
    result = collection.delete_one({"email": email})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="User not found")

    return JSONResponse(status_code=200, content={"status": "success", "message": "Account deleted"})


app.include_router(user_router)
