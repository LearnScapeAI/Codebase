import os
import requests
from dotenv import load_dotenv

load_dotenv()

SERPAPI_KEY = os.getenv("SERPAPI_KEY")

def fetch_real_resources(query: str) -> str:
    if not SERPAPI_KEY:
        return "https://www.kaggle.com/, https://www.coursera.org/, https://www.youtube.com/"

    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google",
        "q": query + " best tutorials",
        "api_key": SERPAPI_KEY,
        "num": 10
    }

    response = requests.get(url, params=params)
    results = response.json()

    links_text = ""
    for result in results.get("organic_results", []):
        title = result.get("title", "")
        link = result.get("link", "")
        links_text += f"{title} - {link}\n"

    return links_text