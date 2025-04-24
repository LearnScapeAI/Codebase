"""
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
"""
"""
import os
import re
import requests
from dotenv import load_dotenv

load_dotenv()

SERPAPI_KEY = os.getenv("SERPAPI_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

TRUSTED_DOMAINS = [
    "youtube.com/watch",
    "coursera.org",
    "kaggle.com",
    "freecodecamp.org",
    "udemy.com",
    "edx.org",
    "geeksforgeeks.org"
]

def extract_youtube_video_id(url: str) -> str:
    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    return match.group(1) if match else None

def is_valid_youtube_video(video_id: str) -> bool:
    url = "https://www.googleapis.com/youtube/v3/videos"
    params = {
        "id": video_id,
        "key": YOUTUBE_API_KEY,
        "part": "status,snippet,contentDetails,statistics"
    }
    response = requests.get(url, params=params)
    data = response.json()
    items = data.get("items", [])
    if not items:
        return False

    video = items[0]
    status = video.get("status", {})
    snippet = video.get("snippet", {})
    stats = video.get("statistics", {})

    return (
        status.get("privacyStatus") == "public" and
        snippet.get("title") and
        int(stats.get("viewCount", 0)) > 1000  # Optional threshold
    )

def fetch_real_resources(query: str) -> str:
    if not SERPAPI_KEY or not YOUTUBE_API_KEY:
        return "https://www.kaggle.com/, https://www.coursera.org/, https://www.youtube.com/"

    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google",
        "q": f"{query} best tutorials",
        "api_key": SERPAPI_KEY,
        "num": 15
    }

    response = requests.get(url, params=params)
    results = response.json()

    links_text = ""
    for result in results.get("organic_results", []):
        title = result.get("title", "")
        link = result.get("link", "")

        # Only allow trusted domains
        if any(domain in link for domain in TRUSTED_DOMAINS):
            if "youtube.com/watch" in link or "youtu.be/" in link:
                video_id = extract_youtube_video_id(link)
                if video_id and is_valid_youtube_video(video_id):
                    links_text += f"{title} - {link}\n"
            else:
                links_text += f"{title} - {link}\n"

    return links_text if links_text else "No valid resources found. Please try a different query."
"""

import os
import re
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import isodate
from datetime import datetime, timedelta

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

TRUSTED_DOMAINS = [
    "youtube.com/watch",
    "coursera.org",
    "kaggle.com",
    "freecodecamp.org",
    "udemy.com",
    "edx.org",
    "geeksforgeeks.org"
]

def google_search(query, api_key, cse_id, num_results=10):
    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'key': api_key,
        'cx': cse_id,
        'q': query + " free",
        'num': num_results,
        'dateRestrict': 'y[1]'  # published within 1 year
    }
    try:
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        return response.json().get("items", [])
    except Exception as e:
        print(f"Error during Google Search API call: {e}")
        return []

def is_valid_url(url):
    try:
        resp = requests.head(url, allow_redirects=True, timeout=5)
        return resp.status_code == 200
    except Exception:
        return False

def get_youtube_video_for_query(query, api_key, max_results=5):
    search_url = "https://www.googleapis.com/youtube/v3/search"
    video_url = "https://www.googleapis.com/youtube/v3/videos"

    def search_youtube(q):
        params = {
            "part": "snippet",
            "q": q + " tutorial",
            "key": api_key,
            "maxResults": max_results,
            "type": "video",
            "publishedAfter": (datetime.now() - timedelta(days=365)).isoformat("T") + "Z"
        }
        try:
            response = requests.get(search_url, params=params)
            response.raise_for_status()
            return response.json().get("items", [])
        except Exception as e:
            print(f"Error during YouTube search API call: {e}")
            return []

    results = search_youtube(query)
    for item in results:
        video_id = item["id"]["videoId"]
        details_params = {
            "part": "contentDetails,snippet",
            "id": video_id,
            "key": api_key
        }
        try:
            details_response = requests.get(video_url, params=details_params)
            details_response.raise_for_status()
            details = details_response.json().get("items", [])
        except Exception as e:
            print(f"Error retrieving video details: {e}")
            continue

        if details:
            duration_str = details[0]["contentDetails"].get("duration", "PT0S")
            published_at = details[0]["snippet"].get("publishedAt", "")
            try:
                duration_seconds = isodate.parse_duration(duration_str).total_seconds()
                published_date = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
            except Exception:
                continue

            if duration_seconds < 600 and published_date > datetime.now() - timedelta(days=365):
                snippet = details[0]["snippet"]
                return {
                    "video_id": video_id,
                    "title": snippet.get("title", "Untitled Video"),
                    "url": f"https://www.youtube.com/watch?v={video_id}"
                }

    return None

def fetch_real_resources(query: str) -> str:
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID or not YOUTUBE_API_KEY:
        return "https://www.kaggle.com/, https://www.coursera.org/, https://www.youtube.com/"

    results = google_search(query, GOOGLE_API_KEY, GOOGLE_CSE_ID, num_results=10)
    resources = ""

    for item in results:
        link = item.get("link", "")
        title = item.get("title", "")
        if any(domain in link for domain in TRUSTED_DOMAINS):
            if is_valid_url(link):
                resources += f"{title} - {link}\n"

    yt_video = get_youtube_video_for_query(query, YOUTUBE_API_KEY)
    if yt_video:
        resources += f"{yt_video['title']} - {yt_video['url']}\n"

    return resources.strip() if resources.strip() else "No valid recent resources found. Try a different topic."