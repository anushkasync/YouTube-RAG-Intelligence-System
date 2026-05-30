import os

import requests
from dotenv import load_dotenv
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi

from logger import get_logger

load_dotenv()

logger = get_logger("TRANSCRIPTS")

SUPADATA_TRANSCRIPT_URL = "https://api.supadata.ai/v1/youtube/transcript"


def extract_video_id(youtube_url: str) -> str:
    parsed_url = urlparse(youtube_url)

    if parsed_url.hostname == "youtu.be":
        return parsed_url.path[1:]

    if parsed_url.hostname in ("www.youtube.com", "youtube.com"):
        if parsed_url.path == "/watch":
            return parse_qs(parsed_url.query)["v"][0]

        if parsed_url.path.startswith("/shorts/"):
            return parsed_url.path.split("/")[2]

    raise ValueError("Invalid YouTube URL")


def normalize_transcript(transcript):
    if hasattr(transcript, "snippets"):
        return " ".join(snippet.text for snippet in transcript.snippets)

    if isinstance(transcript, str):
        return transcript.strip()

    if isinstance(transcript, dict):
        if "content" in transcript:
            return normalize_transcript(transcript["content"])
        return transcript.get("text", "").strip()

    if isinstance(transcript, list):
        parts = []
        for item in transcript:
            if isinstance(item, dict):
                parts.append(item.get("text", ""))
            else:
                parts.append(getattr(item, "text", ""))
        return " ".join(parts).strip()

    raise ValueError(f"Unsupported transcript format: {type(transcript)}")


def _parse_supadata_payload(data):
    for key in ("content", "transcript", "text"):
        if key in data and data[key]:
            return normalize_transcript(data[key])

    raise RuntimeError("Supadata returned empty transcript")


def fetch_from_supadata(video_id: str, language="en"):
    api_key = os.getenv("SUPADATA_API_KEY")

    if not api_key:
        raise RuntimeError("SUPADATA_API_KEY not set")

    response = requests.get(
        SUPADATA_TRANSCRIPT_URL,
        params={
            "videoId": video_id,
            "lang": language,
            "text": "true",
        },
        headers={"x-api-key": api_key},
        timeout=30,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Supadata failed ({response.status_code}): {response.text}"
        )

    text = _parse_supadata_payload(response.json())

    if not text:
        raise RuntimeError("Supadata returned empty transcript")

    return text


def fetch_from_youtube_api(video_id, language="en"):
    yt_api = YouTubeTranscriptApi()
    raw = yt_api.fetch(video_id, languages=[language])
    text = normalize_transcript(raw)

    if not text:
        raise RuntimeError("YouTubeTranscriptApi returned empty transcript")

    return text


def get_transcript(youtube_url: str, language="en"):
    video_id = extract_video_id(youtube_url)

    try:
        text = fetch_from_supadata(video_id, language)
        return {
            "video_id": video_id,
            "source": "supadata",
            "text": text,
        }
    except Exception as e:
        logger.warning(f"Supadata failed: {e}")

    try:
        text = fetch_from_youtube_api(video_id, language)
        return {
            "video_id": video_id,
            "source": "youtube_transcript_api",
            "text": text,
        }
    except Exception as e:
        logger.warning(f"YouTubeTranscriptApi failed: {e}")

    return {
        "video_id": video_id,
        "error": "All transcript sources failed",
    }
