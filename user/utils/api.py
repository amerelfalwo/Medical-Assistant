import requests
import uuid
import os
import sys

# Add confg path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ui.confg import API_URL

SESSION_ID = str(uuid.uuid4())

def upload_pdfs_api(uploaded_files):
    url = f"{API_URL}/upload"
    files = [("files", (file.name, file.getvalue(), "application/pdf")) for file in uploaded_files]
    response = requests.post(url, files=files)
    return response

def ask_question(question):
    url = f"{API_URL}/ask"
    payload = {
        "session_id": SESSION_ID,
        "question": question
    }
    response = requests.post(url, json=payload)
    return response
