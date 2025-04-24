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
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import isodate
from datetime import datetime, timedelta
import time

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# Expanded list of trusted domains with focus on educational content
TRUSTED_DOMAINS = [
    "geeksforgeeks.org",
    "w3schools.com",
    "freecodecamp.org",
    "kaggle.com",
    "tutorialspoint.com",
    "educative.io",
    "javatpoint.com",
    "codecademy.com",
    "programiz.com",
    "khanacademy.org"
]

def google_search(query, api_key, cse_id, num_results=10, site_filter=None):
    """
    Perform a Google search with optional site filtering
    """
    search_url = "https://www.googleapis.com/customsearch/v1"
    
    # Build the query with site filtering if specified
    search_query = query
    if site_filter:
        search_query += f" site:{site_filter}"
    
    params = {
        'key': api_key,
        'cx': cse_id,
        'q': search_query + " tutorial",
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

def is_valid_url(url, timeout=5):
    """
    Check if a URL is valid and accessible
    Uses a more robust validation method
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        resp = requests.head(url, headers=headers, allow_redirects=True, timeout=timeout)
        return 200 <= resp.status_code < 400  # Accept all 2xx and 3xx status codes
    except Exception:
        try:
            # Some sites block HEAD requests, try GET instead
            resp = requests.get(url, headers=headers, stream=True, timeout=timeout)
            resp.close()  # Close connection to avoid reading entire content
            return 200 <= resp.status_code < 400
        except Exception:
            return False

def get_youtube_resource(query, api_key, max_results=8, max_retries=3):
    """
    Get a relevant YouTube tutorial video with improved robustness and retries
    """
    search_url = "https://www.googleapis.com/youtube/v3/search"
    video_url = "https://www.googleapis.com/youtube/v3/videos"

    # Try different query formulations if needed
    query_variations = [
        f"{query} tutorial",
        f"{query} explained",
        f"{query} for beginners",
        f"learn {query}"
    ]
    
    for attempt in range(max_retries):
        # Use a different query variation for each retry
        current_query = query_variations[min(attempt, len(query_variations)-1)]
        
        params = {
            "part": "snippet",
            "q": current_query,
            "key": api_key,
            "maxResults": max_results,
            "type": "video",
            "relevanceLanguage": "en",
            "videoDefinition": "high",
            "publishedAfter": (datetime.now() - timedelta(days=730)).isoformat("T") + "Z"  # 2 years
        }
        
        try:
            response = requests.get(search_url, params=params)
            response.raise_for_status()
            results = response.json().get("items", [])
            
            if not results:
                continue
                
            # Process each result
            for item in results:
                video_id = item["id"]["videoId"]
                video_url_str = f"https://www.youtube.com/watch?v={video_id}"
                
                # Validate the video URL
                if not is_valid_url(video_url_str):
                    continue
                    
                # Get video details
                details_params = {
                    "part": "contentDetails,snippet,statistics",
                    "id": video_id,
                    "key": api_key
                }
                
                details_response = requests.get(video_url, params=details_params)
                details_response.raise_for_status()
                details = details_response.json().get("items", [])
                
                if not details:
                    continue
                    
                video_details = details[0]
                duration_str = video_details["contentDetails"].get("duration", "PT0S")
                
                try:
                    duration_seconds = isodate.parse_duration(duration_str).total_seconds()
                    
                    # Prefer videos between 5 and 20 minutes
                    if 300 <= duration_seconds <= 1200:
                        view_count = int(video_details["statistics"].get("viewCount", 0))
                        # Prefer videos with a decent number of views
                        if view_count > 1000:
                            snippet = video_details["snippet"]
                            return {
                                "type": "video tutorial",
                                "title": snippet.get("title", "YouTube Tutorial"),
                                "url": video_url_str
                            }
                except Exception:
                    continue
            
            # If we've gone through all results and found nothing ideal,
            # return the first valid video as fallback
            for item in results:
                video_id = item["id"]["videoId"]
                video_url_str = f"https://www.youtube.com/watch?v={video_id}"
                if is_valid_url(video_url_str):
                    return {
                        "type": "video tutorial",
                        "title": item["snippet"].get("title", "YouTube Tutorial"),
                        "url": video_url_str
                    }
                    
        except Exception as e:
            print(f"Error in YouTube API call (attempt {attempt+1}): {e}")
            time.sleep(1)  # Prevent rate limiting
    
    # If we still have no results, provide a generic YouTube search URL
    return {
        "type": "video tutorial",
        "title": f"YouTube tutorials for {query}",
        "url": f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
    }

def get_website_resource(query, api_key, cse_id):
    """
    Get a high-quality website resource for concept revision
    """
    # Try to find resources from trusted domains first
    for domain in TRUSTED_DOMAINS:
        results = google_search(f"{query} concept", api_key, cse_id, num_results=5, site_filter=domain)
        
        for item in results:
            link = item.get("link", "")
            title = item.get("title", "")
            
            if is_valid_url(link):
                return {
                    "type": "documentation",
                    "title": title,
                    "url": link
                }
    
    # If no trusted domain resources found, try a general search
    results = google_search(f"{query} concept explanation", api_key, cse_id, num_results=10)
    
    for item in results:
        link = item.get("link", "")
        title = item.get("title", "")
        
        # Make sure it's from a somewhat trustworthy domain
        if any(domain in link.lower() for domain in TRUSTED_DOMAINS):
            if is_valid_url(link):
                return {
                    "type": "documentation",
                    "title": title,
                    "url": link
                }
    
    # Last resort - accept any valid educational resource
    for item in results:
        link = item.get("link", "")
        title = item.get("title", "")
        
        # Skip obvious non-educational resources
        if any(x in link.lower() for x in [".gov", ".org", ".edu"]):
            if is_valid_url(link):
                return {
                    "type": "documentation",
                    "title": title,
                    "url": link
                }
    
    # Fallback to a reliable general resource
    return {
        "type": "documentation",
        "title": f"{query} on GeeksforGeeks",
        "url": f"https://www.geeksforgeeks.org/search?q={query.replace(' ', '+')}"
    }

def fetch_real_resources(query: str) -> str:
    """
    Main function to fetch exactly two resources: one video tutorial and one documentation resource
    """
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID or not YOUTUBE_API_KEY:
        return "API keys not configured. Please set GOOGLE_API_KEY, GOOGLE_CSE_ID, and YOUTUBE_API_KEY in your .env file."

    # Get a YouTube video tutorial
    video_resource = get_youtube_resource(query, YOUTUBE_API_KEY)
    
    # Get a website documentation/concept resource
    website_resource = get_website_resource(query, GOOGLE_API_KEY, GOOGLE_CSE_ID)
    
    # Format the output
    resources = f"Resources for '{query}':\n\n"
    
    if video_resource:
        resources += f"📺 {video_resource['title']}\n   Link: {video_resource['url']}\n\n"
    
    if website_resource:
        resources += f"📚 {website_resource['title']}\n   Link: {website_resource['url']}"
    
    return resources.strip()

# Example usage
if __name__ == "__main__":
    topic = input("Enter a learning topic: ")
    print("\nFinding resources...\n")
    results = fetch_real_resources(topic)
    print(results)