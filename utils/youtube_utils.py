import requests
import pandas as pd
import re
from utils.text_utils import predict_text


def search_youtube_videos(api_key, query, max_results=5):
    endpoint = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": max_results,
        "key": api_key
    }

    response = requests.get(endpoint, params=params)
    if response.status_code != 200:
        return [], f"API Error: {response.status_code}"

    items = response.json().get("items", [])
    video_info = [
        {
            "video_id": item["id"]["videoId"],
            "title": item["snippet"]["title"],
            "thumbnail": item["snippet"]["thumbnails"]["default"]["url"]
        }
        for item in items if item["id"]["kind"] == "youtube#video"
    ]

    return video_info, None


def fetch_youtube_comments(api_key, video_url, max_comments=20):
    video_id_match = re.search(r"v=([a-zA-Z0-9_-]{11})", video_url)
    if not video_id_match:
        return None, "Invalid YouTube URL"

    video_id = video_id_match.group(1)

    endpoint = "https://www.googleapis.com/youtube/v3/commentThreads"
    comments = []
    params = {
        "part": "snippet",
        "videoId": video_id,
        "maxResults": 100,
        "textFormat": "plainText",
        "key": api_key
    }

    response = requests.get(endpoint, params=params)
    if response.status_code != 200:
        return None, f"Error fetching comments: {response.status_code}"

    data = response.json()
    for item in data.get("items", []):
        comment_text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        prediction = predict_text(comment_text)
        comments.append({"Comment": comment_text, "Prediction": prediction})

        if len(comments) >= max_comments:
            break

    df = pd.DataFrame(comments)
    return df, None
