from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional

class User(BaseModel):
    full_name: str
    username: str
    email: EmailStr
    password: Optional[str] = None      # optional for Google users
    is_google_user: bool = False        # flag to mark Google users
    created_at: Optional[datetime] = None
class LoginUser(BaseModel):
    email: EmailStr
    password: str
