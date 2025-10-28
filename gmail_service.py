import os
import base64
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
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
        scopes=SCOPES,
    )
    return build("gmail", "v1", credentials=creds)

def send_email(to_email: str, subject: str, body: str, attachment_path: str | None = None):
    """
    Send email using Gmail API with optional attachment.
    """
    try:
        service = get_gmail_service()
        message = MIMEMultipart()
        message["to"] = to_email
        message["from"] = MAIL_USERNAME
        message["subject"] = subject

        # Body
        message.attach(MIMEText(body, "html"))

        # Optional attachment
        if attachment_path and os.path.exists(attachment_path):
            part = MIMEBase("application", "octet-stream")
            with open(attachment_path, "rb") as f:
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(attachment_path)}")
            message.attach(part)

        # Encode message
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        send_message = {"raw": raw_message}

        sent = service.users().messages().send(userId="me", body=send_message).execute()
        print(f"✅ Email sent to {to_email} (Gmail message ID: {sent['id']})")

    except Exception as e:
        print("❌ Gmail send failed:", str(e))
        raise
