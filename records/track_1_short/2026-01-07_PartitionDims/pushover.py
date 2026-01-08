import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_TOKEN = os.getenv("API_TOKEN")
USER_KEY = os.getenv("USER_KEY")

def send_pushover(message: str, title: str = "Training Update", priority: int = 1) -> bool:
    """Send a pushover notification. Returns True if successful."""
    payload = {
        "token": API_TOKEN,
        "user": USER_KEY,
        "device": "iphone",
        "message": message,
        "title": title,
        "priority": priority
    }
    
    response = requests.post("https://api.pushover.net/1/messages.json", data=payload)
    result = response.json()
    return result.get("status") == 1

if __name__ == "__main__":
    send_pushover("Hello iPhone", "Training Update", 1)