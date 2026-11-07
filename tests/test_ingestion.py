"""Unit tests for WordPress API client with mocking."""

import pytest
from unittest.mock import patch, MagicMock
from ingestion.wordpress import WordPressClient, Article
from utils.exceptions import IngestionError

def test_fetch_articles_success():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.is_redirect = False
    mock_response.headers = {}
    mock_response.json.return_value = [
        {
            "id": 101,
            "title": {"rendered": "AI Breakthrough in Reasoning"},
            "content": {"rendered": "<p>A new model demonstrates step-by-step verifiable logic.</p>"},
            "link": "https://techcrunch.com/article-101",
            "date": "2026-11-01T10:00:00",
        },
        {
            "id": 102,
            "title": {"rendered": "Vector Databases Scale Up"},
            "content": {"rendered": "<p>FAISS flat inner product provides fast exact similarity search.</p>"},
            "link": "https://techcrunch.com/article-102",
            "date": "2026-11-02T11:00:00",
        },
    ]

    with patch.object(WordPressClient, "_safe_get", return_value=mock_response):
        # Using a valid public domain to pass validation
        client = WordPressClient(base_url="https://techcrunch.com")
        articles = client.fetch_articles(max_articles=2, per_page=10)

        assert len(articles) == 2
        assert articles[0].post_id == 101
        assert "AI Breakthrough" in articles[0].title
        assert "step-by-step" in articles[0].content
        assert articles[1].post_id == 102

def test_fetch_articles_handles_empty_response():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.is_redirect = False
    mock_response.headers = {}
    mock_response.json.return_value = []

    with patch.object(WordPressClient, "_safe_get", return_value=mock_response):
        client = WordPressClient(base_url="https://techcrunch.com")
        with pytest.raises(IngestionError):
            client.fetch_articles(max_articles=5)

