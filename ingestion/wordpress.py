"""WordPress REST API ingestion client with SSRF safeguards, retries, and pagination."""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set
import requests

from ingestion.preprocessing import clean_html_content, sanitize_text
from ingestion.url_validator import validate_target_url
from utils.exceptions import IngestionError, SSRFSecurityError
from utils.logging_config import logger

@dataclass
class Article:
    """Represents an ingested WordPress article."""
    post_id: int
    title: str
    content: str
    url: str
    date: str
    source: str

class WordPressClient:
    """Fetches public articles via the WordPress REST API with SSRF controls and retries."""

    def __init__(
        self,
        base_url: str,
        timeout: int = 20,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
        max_response_bytes: int = 15 * 1024 * 1024,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.max_response_bytes = max_response_bytes

        # Validate base URL against SSRF
        validate_target_url(self.base_url)

    def _safe_get(self, url: str, params: Optional[Dict] = None) -> requests.Response:
        """Issues an HTTP GET request with SSRF validation, redirects check, and retries."""
        validate_target_url(url)
        session = requests.Session()
        session.max_redirects = 5

        # Custom redirect handling to prevent redirect-based SSRF
        current_url = url
        for attempt in range(self.max_retries):
            try:
                # We disable automatic redirect to re-validate each hop
                response = session.get(
                    current_url,
                    params=params if attempt == 0 else None,
                    timeout=self.timeout,
                    allow_redirects=False,
                    headers={"User-Agent": "RAG-COT-Model-Modernized/1.0", "Accept": "application/json"},
                )

                if response.is_redirect or response.status_code in (301, 302, 307, 308):
                    redirect_target = response.headers.get("Location")
                    if not redirect_target:
                        raise IngestionError("Redirect response missing Location header.")
                    # Re-validate redirect target against SSRF
                    validate_target_url(redirect_target)
                    current_url = redirect_target
                    continue

                if response.status_code == 429:
                    wait_time = float(response.headers.get("Retry-After", self.backoff_factor * (2 ** attempt)))
                    logger.warning(f"Rate limited (429) fetching {url}. Backing off for {wait_time:.1f}s.")
                    time.sleep(min(wait_time, 10.0))
                    continue

                if response.status_code >= 500:
                    wait_time = self.backoff_factor * (2 ** attempt)
                    logger.warning(f"Server error ({response.status_code}). Retry {attempt + 1}/{self.max_retries}...")
                    time.sleep(wait_time)
                    continue

                # Enforce response body size limit
                content_len = response.headers.get("Content-Length")
                if content_len and int(content_len) > self.max_response_bytes:
                    raise IngestionError(f"Response size {content_len} bytes exceeds limit of {self.max_response_bytes} bytes.")

                return response

            except (requests.Timeout, requests.ConnectionError) as e:
                wait_time = self.backoff_factor * (2 ** attempt)
                logger.warning(f"Network error ({e}) on attempt {attempt + 1}. Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)

        raise IngestionError(f"Failed to fetch {url} after {self.max_retries} attempts.")

    def fetch_articles(
        self,
        max_articles: int = 50,
        per_page: int = 20,
    ) -> List[Article]:
        """
        Fetches articles across pages until max_articles is reached or pages are exhausted.
        """
        endpoint = f"{self.base_url}/wp-json/wp/v2/posts"
        articles: List[Article] = []
        seen_ids: Set[int] = set()

        page = 1
        per_page = min(max(1, per_page), 100)

        while len(articles) < max_articles:
            fetch_count = min(per_page, max_articles - len(articles))
            params = {
                "page": page,
                "per_page": fetch_count,
                "_fields": "id,link,title,content,date",
            }

            try:
                response = self._safe_get(endpoint, params=params)
            except Exception as e:
                if page > 1:
                    logger.info(f"Finished ingestion on page {page}: {e}")
                    break
                raise IngestionError(f"Failed initial connection to WordPress endpoint {endpoint}: {e}")

            if response.status_code == 400 and page > 1:
                # WordPress returns 400 when page number exceeds available pages
                break

            if response.status_code != 200:
                raise IngestionError(f"HTTP error {response.status_code} from WordPress REST API: {response.text[:200]}")

            try:
                batch = response.json()
            except Exception as e:
                raise IngestionError(f"Malformed JSON response from WordPress API: {e}")

            if not isinstance(batch, list) or not batch:
                break

            for item in batch:
                if not isinstance(item, dict):
                    continue
                try:
                    pid = int(item.get("id", 0))
                    if pid <= 0 or pid in seen_ids:
                        continue
                    seen_ids.add(pid)

                    raw_title = item.get("title", {}).get("rendered", "") if isinstance(item.get("title"), dict) else str(item.get("title", ""))
                    raw_content = item.get("content", {}).get("rendered", "") if isinstance(item.get("content"), dict) else str(item.get("content", ""))
                    post_url = item.get("link", self.base_url)
                    date_str = item.get("date", "")

                    title = sanitize_text(raw_title) or f"Article #{pid}"
                    cleaned_body = clean_html_content(raw_content)

                    # Only keep article if it has some readable text
                    if not cleaned_body:
                        cleaned_body = title

                    articles.append(
                        Article(
                            post_id=pid,
                            title=title,
                            content=cleaned_body,
                            url=post_url,
                            date=date_str,
                            source=self.base_url,
                        )
                    )

                    if len(articles) >= max_articles:
                        break
                except Exception as err:
                    logger.warning(f"Skipping malformed post item: {err}")

            if len(batch) < fetch_count:
                # End of posts
                break

            page += 1

        if not articles:
            raise IngestionError(f"No valid articles found at {endpoint}. Ensure the WordPress REST API is public.")

        logger.info(f"Successfully ingested {len(articles)} articles from {self.base_url}")
        return articles

