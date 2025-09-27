import os
import base64
from email.mime.text import MIMEText
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

MAIL_USERNAME = os.getenv("MAIL_USERNAME")
CLIENT_ID = os.getenv("GMAIL_CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
REFRESH_TOKEN = os.getenv("OAUTH2_REFRESH_TOKEN")

SCOPES = ["https://www.googleapis.com/auth/gmail.send"]

def get_gmail_service():
    creds = Credentials(
        token=None,
        refresh_token=REFRESH_TOKEN,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        scopes=SCOPES
    )
    service = build("gmail", "v1", credentials=creds)
    return service

def send_email(to_email: str, subject: str, body: str):
    service = get_gmail_service()
    message = MIMEText(body, "html")
    message["to"] = to_email
    message["from"] = MAIL_USERNAME
    message["subject"] = subject

    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    sent = service.users().messages().send(userId="me", body={"raw": raw}).execute()
    print("✅ Email sent! Gmail response ID:", sent["id"])
