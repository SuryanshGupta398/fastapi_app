from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional

class User(BaseModel):
    full_name: str
    username: str
    email: EmailStr
    password: str   # in real apps, store hashed password
    created_at: Optional[datetime] = None
