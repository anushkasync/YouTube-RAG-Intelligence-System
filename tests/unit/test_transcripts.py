import pytest

from transcripts import extract_video_id, normalize_transcript


def test_extract_video_id_watch():
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    assert extract_video_id(url) == "dQw4w9WgXcQ"


def test_extract_video_id_short_url():
    url = "https://youtu.be/dQw4w9WgXcQ"
    assert extract_video_id(url) == "dQw4w9WgXcQ"


def test_normalize_string():
    assert normalize_transcript("  hello world  ") == "hello world"


def test_normalize_supadata_content_list():
    payload = [
        {"text": "Hello", "offset": 0, "duration": 1},
        {"text": "world", "offset": 1, "duration": 1},
    ]
    assert normalize_transcript(payload) == "Hello world"


def test_normalize_supadata_content_dict():
    assert normalize_transcript({"content": "plain text"}) == "plain text"
