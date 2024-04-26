import os
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from Google import Create_Service 

client_id = "1061625394649-0cfjjlnjmorlultv7jc8ot8jmcifu6st.apps.googleusercontent.com"
client_secret = "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"


# Access variables
CLIENT_ID = client_id
CLIENT_SECRET = client_secret

SCOPES = ['https://www.googleapis.com/auth/drive']

# Create flow object using client ID and client secret
service = Create_Service(CLIENT_ID, CLIENT_SECRET, SCOPES)

print(dir(service))