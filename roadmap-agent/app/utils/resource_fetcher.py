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
import os
import requests
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Tuple
import json
import re
from urllib.parse import quote
import time
import random

load_dotenv()

SERPAPI_KEY = os.getenv("SERPAPI_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

class ResourceValidator:
    """Class to validate and verify educational resources"""

    @staticmethod
    def is_valid_url(url: str) -> bool:
        """Basic validation to ensure URL is properly formatted"""
        pattern = re.compile(
            r'^(?:http|https)://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain
            r'localhost|'  # localhost
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # IP
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        return bool(pattern.match(url))

    @staticmethod
    def verify_resource_exists(url: str) -> bool:
        """Verify that a URL actually exists by checking HTTP status"""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.head(url, headers=headers, timeout=5, allow_redirects=True)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    @staticmethod
    def is_educational_domain(url: str) -> bool:
        """Check if the URL is from a known educational domain"""
        educational_domains = [
            'khanacademy.org', 'coursera.org', 'edx.org', 'udemy.com',
            'ocw.mit.edu', 'nptel.ac.in', 'youtube.com', 'freecodecamp.org',
            'w3schools.com', 'tutorialspoint.com', 'geeksforgeeks.org',
            'javatpoint.com', 'stackoverflow.com', 'gateoverflow.in',
            'gate.iitd.ac.in', 'madeeasy.in', 'physicswallahalakhpandey.com',
            'physicswallah.com', 'vedantu.com', 'unacademy.com', 'byju.com',
            'madeeasypublications.org', 'nptel.ac.in', 'ncert.nic.in',
            'kaggle.com', 'towardsdatascience.com', 'github.com', 'docs.python.org',
            'tensorflow.org', 'pytorch.org', 'scikit-learn.org', 'pandas.pydata.org'
        ]
        return any(domain in url.lower() for domain in educational_domains)

class ResourceFetcher:
    """Class for fetching educational resources"""

    def __init__(self):
        self.validator = ResourceValidator()
        self.google_api_key = SERPAPI_KEY
        self.youtube_api_key = YOUTUBE_API_KEY

    def fetch_google_results(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
        """Fetch verified resources from Google using SerpAPI"""
        if not self.google_api_key:
            print("No SERPAPI_KEY found in environment")
            return []

        url = "https://serpapi.com/search.json"
        params = {
            "engine": "google",
            "q": query + " tutorial guide resources",
            "api_key": self.google_api_key,
            "num": num_results * 3  # Fetch more to account for filtering
        }

        try:
            response = requests.get(url, params=params)
            if response.status_code != 200:
                print(f"SerpAPI error: {response.status_code} - {response.text}")
                return []

            results = response.json()

            formatted_results = []
            for result in results.get("organic_results", []):
                url = result.get("link", "")
                if (self.validator.is_valid_url(url) and
                        not any(x == url.lower() for x in ["pinterest.com", "instagram.com", "facebook.com"]) and
                        (self.validator.is_educational_domain(url) or self.validator.verify_resource_exists(url))):

                    formatted_results.append({
                        "title": result.get("title", ""),
                        "link": url,
                        "source": "Google",
                        "description": result.get("snippet", "")[:100] + "..." if result.get("snippet") else ""
                    })

                    if len(formatted_results) >= num_results:
                        break

            return formatted_results
        except requests.exceptions.RequestException as e:
            print(f"Error fetching Google results: {e}")
            return []

    def fetch_youtube_results(self, query: str, num_results: int = 3) -> List[Dict[str, str]]:
        """Fetch educational videos from YouTube API with validation"""
        if not self.youtube_api_key:
            print("No YOUTUBE_API_KEY found in environment")
            return []

        url = "https://www.googleapis.com/youtube/v3/search"

        # Add educational terms to the query
        educational_query = query
        if "gate" in query.lower():
            educational_query += " gate exam tutorial"
        elif "jee" in query.lower():
            educational_query += " jee preparation tutorial"
        else:
            educational_query += " tutorial"

        params = {
            "part": "snippet",
            "q": educational_query,
            "key": self.youtube_api_key,
            "maxResults": num_results * 3,  # Fetch more to account for filtering
            "type": "video",
            "relevanceLanguage": "en",
            "videoEmbeddable": "true"
        }

        try:
            response = requests.get(url, params=params)
            if response.status_code != 200:
                print(f"YouTube API error: {response.status_code} - {response.text}")
                return []

            results = response.json()

            # Get video IDs
            video_ids = [item["id"]["videoId"] for item in results.get("items", [])
                         if "videoId" in item.get("id", {})]

            if not video_ids:
                return []

            # Get video statistics to help filter for quality content
            stats_url = "https://www.googleapis.com/youtube/v3/videos"
            stats_params = {
                "part": "statistics,contentDetails",
                "id": ",".join(video_ids),
                "key": self.youtube_api_key
            }

            stats_response = requests.get(stats_url, params=stats_params)
            if stats_response.status_code != 200:
                print(f"YouTube Stats API error: {stats_response.status_code}")
                stats_data = {}
            else:
                stats_data = {item["id"]: item for item in stats_response.json().get("items", [])}

            formatted_results = []
            for item in results.get("items", []):
                video_id = item.get("id", {}).get("videoId", "")
                if video_id:
                    stats = stats_data.get(video_id, {}).get("statistics", {})
                    view_count = int(stats.get("viewCount", 0))

                    # Only include videos with good engagement
                    if view_count > 1000:
                        video_url = f"https://www.youtube.com/watch?v={video_id}"
                        # Verify this is a real working URL
                        if self.validator.verify_resource_exists(video_url):
                            formatted_results.append({
                                "title": item.get("snippet", {}).get("title", ""),
                                "link": video_url,
                                "source": "YouTube",
                                "views": view_count
                            })

                    if len(formatted_results) >= num_results:
                        break

            return formatted_results
        except requests.exceptions.RequestException as e:
            print(f"Error fetching YouTube results: {e}")
            return []

    def fetch_specialized_resources(self, topic: str) -> List[Dict[str, str]]:
        """Fetch specialized resources based on the topic"""

        # Map topics to specialized resources
        topic_resources = {
            "gate": [
                {"title": "GATE Overflow", "link": "https://gateoverflow.in/", "source": "GATE Overflow"},
                {"title": "NPTEL GATE Lectures", "link": "https://nptel.ac.in/courses/gate", "source": "NPTEL"},
                {"title": "Made Easy GATE Resources", "link": "https://www.madeeasy.in/home/GATE", "source": "Made Easy"}
            ],
            "jee": [
                {"title": "Physics Wallah - JEE", "link": "https://www.physicswallah.com/jee-main", "source": "Physics Wallah"},
                {"title": "NCERT Books for JEE", "link": "https://ncert.nic.in/textbook.php", "source": "NCERT"},
                {"title": "Unacademy JEE", "link": "https://unacademy.com/goal/jee-main-and-advanced", "source": "Unacademy"}
            ],
            "data science": [
                {"title": "Kaggle Learn", "link": "https://www.kaggle.com/learn", "source": "Kaggle"},
                {"title": "DataCamp Community", "link": "https://www.datacamp.com/community/tutorials", "source": "DataCamp"},
                {"title": "Towards Data Science", "link": "https://towardsdatascience.com/", "source": "Medium"}
            ],
            "machine learning": [
                {"title": "Google ML Crash Course", "link": "https://developers.google.com/machine-learning/crash-course", "source": "Google"},
                {"title": "Andrew Ng's ML Course", "link": "https://www.coursera.org/learn/machine-learning", "source": "Coursera"},
                {"title": "ML Mastery", "link": "https://machinelearningmastery.com/start-here/", "source": "ML Mastery"}
            ],
            "python": [
                {"title": "Real Python Tutorials", "link": "https://realpython.com/", "source": "Real Python"},
                {"title": "Python.org Documentation", "link": "https://docs.python.org/3/tutorial/", "source": "Python.org"},
                {"title": "W3Schools Python", "link": "https://www.w3schools.com/python/", "source": "W3Schools"}
            ],
            "electronics": [
                {"title": "All About Circuits", "link": "https://www.allaboutcircuits.com/textbook/", "source": "All About Circuits"},
                {"title": "Electronics Tutorials", "link": "https://www.electronics-tutorials.ws/", "source": "Electronics Tutorials"},
                {"title": "Circuit Digest", "link": "https://circuitdigest.com/tutorials", "source": "Circuit Digest"}
            ]
        }

        # Find which topic categories apply
        applicable_topics = []
        for key in topic_resources:
            if key in topic.lower():
                applicable_topics.append(key)

        # Get resources from all applicable topics
        resources = []
        for applicable_topic in applicable_topics:
            resources.extend(topic_resources[applicable_topic])

        # Verify all resources exist
        verified_resources = []
        for resource in resources:
            if self.validator.verify_resource_exists(resource["link"]):
                verified_resources.append(resource)

        return verified_resources[:3]  # Return top 3 specialized resources

    def fetch_verified_resources(self, query: str, max_results: int = 10) -> List[Dict[str, str]]:
        """Fetch and verify resources for the given query"""

        # Define default resources (guaranteed to work)
        default_resources = [
            {"title": "Khan Academy", "link": "https://www.khanacademy.org/", "source": "Khan Academy"},
            {"title": "Coursera - Free Courses", "link": "https://www.coursera.org/courses?query=free", "source": "Coursera"},
            {"title": "edX - Free Courses", "link": "https://www.edx.org/search?q=free", "source": "edX"},
            {"title": "MIT OpenCourseWare", "link": "https://ocw.mit.edu/", "source": "MIT"},
            {"title": "freeCodeCamp", "link": "https://www.freecodecamp.org/", "source": "freeCodeCamp"}
        ]

        all_resources = []

        # Try specialized resources first
        specialized_resources = self.fetch_specialized_resources(query)
        if specialized_resources:
            all_resources.extend(specialized_resources)

        # Try YouTube next (most reliable for educational content)
        youtube_resources = self.fetch_youtube_results(query, 3)
        if youtube_resources:
            all_resources.extend(youtube_resources)

        # Add Google search results
        google_resources = self.fetch_google_results(query, 4)
        if google_resources:
            all_resources.extend(google_resources)

        # If we still don't have enough, add some default resources
        if len(all_resources) < max_results:
            for resource in default_resources:
                if len(all_resources) >= max_results:
                    break
                # Add if not already in results
                if not any(resource["link"] == existing["link"] for existing in all_resources):
                    all_resources.append(resource)

        return all_resources[:max_results]

def fetch_real_resources(query: str) -> str:
    """Fetch verified educational resources and return in the required format"""

    fetcher = ResourceFetcher()
    resources = fetcher.fetch_verified_resources(query, 5)

    # Format as required by the existing structure
    links_text = ""
    for resource in resources:
        title = resource.get("title", "Educational Resource")
        link = resource.get("link", "")
        links_text += f"{title} - {link}\n"

    # Ensure we return at least something
    if not links_text:
        links_text = "Khan Academy - https://www.khanacademy.org/\nW3Schools - https://www.w3schools.com/"

    return links_text

# Example usage
if __name__ == "__main__":
    test_query = "GATE ECE circuits"
    result = fetch_real_resources(test_query)
    print(result)