import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
USERNAME = os.getenv("USERNAME")

# Get the date range
since_date = (datetime.utcnow() - timedelta(days=10)).isoformat()

headers = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

url = f"https://api.github.com/users/{USERNAME}/events"
response = requests.get(url, headers=headers)
events = response.json()

for event in events:
    created_at = event["created_at"]
    if created_at > since_date:
        event_type = event["type"]
        repo_name = event["repo"]["name"]
        print(f"\n📅 {created_at} - 🧩 {event_type} - 📦 {repo_name}")
        
        if event_type == "PushEvent":
            for commit in event["payload"]["commits"]:
                print(f"  🔗 Commit: {commit['sha'][:7]}")
                print(f"  ✍️  Message: {commit['message']}")
                print(f"  👤 Author: {commit['author']['name']}")