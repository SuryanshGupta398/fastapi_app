import os
import base64
from email.mime.text import MIMEText
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
REFRESH_TOKEN = os.getenv("OAUTH2_REFRESH_TOKEN")
MAIL_USERNAME = os.getenv("MAIL_USERNAME")  # Gmail address

def get_gmail_service():
    creds = Credentials(
        token=None,
        refresh_token=REFRESH_TOKEN,
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        token_uri="https://oauth2.googleapis.com/token"
    )
    if creds.expired or not creds.valid:
        creds.refresh(Request())
    service = build('gmail', 'v1', credentials=creds)
    return service

def send_email(to_email: str, subject: str, body: str):
    service = get_gmail_service()
    message = MIMEText(body, "html")
    message['to'] = to_email
    message['from'] = MAIL_USERNAME
    message['subject'] = subject
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    service.users().messages().send(userId='me', body={'raw': raw}).execute()
    print(f"✅ Email sent to {to_email}")
