# pipelines/upload_to_drive.py
# Uploads FCI_Device_Health_Export.csv to Google Drive

import os
import json
from pathlib import Path
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# ====================================================
# CONFIGURATION
# ====================================================

FOLDER_ID = "18GRxAym5XHIFiOeIkDHgygDu8eLnQ9aQ"
EXPORT_FILE = Path(__file__).resolve().parent.parent / "powerbi_exports" / "FCI_Device_Health_Export.csv"
DRIVE_FILENAME = "FCI_Device_Health_Export.csv"

def get_drive_service():
    """Authenticate and return Google Drive service"""
    creds_json = os.environ.get("GOOGLE_CREDENTIALS")
    if not creds_json:
        raise ValueError("GOOGLE_CREDENTIALS environment variable not set")
    
    creds_dict = json.loads(creds_json)
    creds = service_account.Credentials.from_service_account_info(
        creds_dict,
        scopes=["https://www.googleapis.com/auth/drive"]
    )
    return build("drive", "v3", credentials=creds)

def get_existing_file_id(service, filename, folder_id):
    """Check if file already exists in the folder"""
    query = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get("files", [])
    return files[0]["id"] if files else None

def upload_to_drive():
    """Upload or update the CSV file in Google Drive"""
    print("=" * 60)
    print(" UPLOADING TO GOOGLE DRIVE")
    print("=" * 60)

    if not EXPORT_FILE.exists():
        raise FileNotFoundError(f"Export file not found: {EXPORT_FILE}")

    print(f"\n Authenticating with Google Drive...")
    service = get_drive_service()

    media = MediaFileUpload(
        str(EXPORT_FILE),
        mimetype="text/csv",
        resumable=True
    )

    existing_id = get_existing_file_id(service, DRIVE_FILENAME, FOLDER_ID)

    if existing_id:
        print(f" File already exists - updating in place...")
        file = service.files().update(
            fileId=existing_id,
            media_body=media
        ).execute()
        print(f" Updated: {file.get('name')} (ID: {file.get('id')})")
    else:
        print(f" Uploading new file...")
        file_metadata = {
            "name": DRIVE_FILENAME,
            "parents": [FOLDER_ID]
        }
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields="id, name"
        ).execute()
        print(f" Uploaded: {file.get('name')} (ID: {file.get('id')})")

    print(f"\n Successfully uploaded to Google Drive folder:")
    print(f"   https://drive.google.com/drive/folders/{FOLDER_ID}")
    print("=" * 60)

if __name__ == "__main__":
    upload_to_drive()
