"""Commentary collection service for Bible-AI."""

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class CommentarySource:
    """Configuration for a commentary source."""

    name: str
    url: str
    type: str  # 'api', 'web', 'local'
    format: str  # 'json', 'xml', 'html', 'text'
    authentication: Optional[Dict[str, str]] = None
    parser_config: Optional[Dict[str, Any]] = None


class CommentaryCollector:
    """Service for collecting biblical commentaries from various sources."""

    def __init__(self, config_path: str = "config/data_config.json"):
        self.config_path = config_path
        self.load_config()
        self.setup_directories()

    def load_config(self):
        """Load configuration from file."""
        try:
            with open(self.config_path, "r") as f:
                self.config = json.load(f)

            # Load commentary sources
            sources_path = os.path.join(
                os.path.dirname(self.config_path), "commentary_sources.json"
            )
            with open(sources_path, "r") as f:
                sources_config = json.load(f)
                self.sources = [
                    CommentarySource(**source) for source in sources_config["sources"]
                ]
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise

    def setup_directories(self):
        """Create necessary directories."""
        base_dir = self.config.get("data_dir", "data")
        self.raw_dir = os.path.join(base_dir, "raw", "commentaries")
        self.processed_dir = os.path.join(base_dir, "processed", "commentaries")

        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

    async def collect_from_api(self, source: CommentarySource) -> List[Dict[str, Any]]:
        """Collect commentary data from an API source."""
        async with aiohttp.ClientSession() as session:
            headers = source.authentication or {}
            async with session.get(source.url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._process_api_response(data, source)
                else:
                    logger.error(
                        f"API request failed for {source.name}: {response.status}"
                    )
                    return []

    async def collect_from_web(self, source: CommentarySource) -> List[Dict[str, Any]]:
        """Collect commentary data by scraping web pages."""
        async with aiohttp.ClientSession() as session:
            async with session.get(source.url) as response:
                if response.status == 200:
                    html = await response.text()
                    return self._process_web_content(html, source)
                else:
                    logger.error(
                        f"Web scraping failed for {source.name}: {response.status}"
                    )
                    return []

    def collect_from_local(self, source: CommentarySource) -> List[Dict[str, Any]]:
        """Collect commentary data from local files."""
        try:
            file_path = os.path.join(
                self.raw_dir, source.url
            )  # url field contains local path
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            if source.format == "json":
                return json.loads(content)
            elif source.format == "xml":
                return self._process_xml_content(content, source)
            else:
                return self._process_text_content(content, source)
        except Exception as e:
            logger.error(f"Error processing local file for {source.name}: {e}")
            return []

    def _process_api_response(
        self, data: Dict[str, Any], source: CommentarySource
    ) -> List[Dict[str, Any]]:
        """Process API response data."""
        entries = []
        parser_config = source.parser_config or {}

        try:
            # Extract entries based on parser configuration
            entry_path = parser_config.get("entry_path", "entries")
            content_key = parser_config.get("content_key", "content")
            ref_key = parser_config.get("reference_key", "reference")

            raw_entries = data
            for path_part in entry_path.split("."):
                raw_entries = raw_entries.get(path_part, [])

            for entry in raw_entries:
                processed_entry = {
                    "source": source.name,
                    "collected_at": datetime.now().isoformat(),
                    "content": entry.get(content_key, ""),
                    "reference": entry.get(ref_key, ""),
                }
                entries.append(processed_entry)

        except Exception as e:
            logger.error(f"Error processing API response from {source.name}: {e}")

        return entries

    def _process_web_content(
        self, html: str, source: CommentarySource
    ) -> List[Dict[str, Any]]:
        """Process scraped web content."""
        entries = []
        parser_config = source.parser_config or {}

        try:
            soup = BeautifulSoup(html, "html.parser")
            entry_selector = parser_config.get("entry_selector", "div.commentary-entry")
            content_selector = parser_config.get("content_selector", "div.content")
            ref_selector = parser_config.get("reference_selector", "div.reference")

            for entry_elem in soup.select(entry_selector):
                content = entry_elem.select_one(content_selector)
                reference = entry_elem.select_one(ref_selector)

                if content:
                    entries.append(
                        {
                            "source": source.name,
                            "collected_at": datetime.now().isoformat(),
                            "content": content.get_text(strip=True),
                            "reference": (
                                reference.get_text(strip=True) if reference else ""
                            ),
                        }
                    )

        except Exception as e:
            logger.error(f"Error processing web content from {source.name}: {e}")

        return entries

    def _process_xml_content(
        self, content: str, source: CommentarySource
    ) -> List[Dict[str, Any]]:
        """Process XML content."""
        entries = []
        parser_config = source.parser_config or {}

        try:
            soup = BeautifulSoup(content, "xml")
            entry_tag = parser_config.get("entry_tag", "entry")
            content_tag = parser_config.get("content_tag", "content")
            ref_tag = parser_config.get("reference_tag", "reference")

            for entry_elem in soup.find_all(entry_tag):
                content = entry_elem.find(content_tag)
                reference = entry_elem.find(ref_tag)

                if content:
                    entries.append(
                        {
                            "source": source.name,
                            "collected_at": datetime.now().isoformat(),
                            "content": content.get_text(strip=True),
                            "reference": (
                                reference.get_text(strip=True) if reference else ""
                            ),
                        }
                    )

        except Exception as e:
            logger.error(f"Error processing XML content from {source.name}: {e}")

        return entries

    def _process_text_content(
        self, content: str, source: CommentarySource
    ) -> List[Dict[str, Any]]:
        """Process plain text content."""
        entries = []
        parser_config = source.parser_config or {}

        try:
            # Split content into entries based on configured delimiter
            delimiter = parser_config.get("entry_delimiter", "\n\n")
            raw_entries = content.split(delimiter)

            for raw_entry in raw_entries:
                if not raw_entry.strip():
                    continue

                # Try to extract reference and content
                parts = raw_entry.split("\n", 1)
                if len(parts) == 2:
                    reference, content = parts
                else:
                    reference, content = "", parts[0]

                entries.append(
                    {
                        "source": source.name,
                        "collected_at": datetime.now().isoformat(),
                        "content": content.strip(),
                        "reference": reference.strip(),
                    }
                )

        except Exception as e:
            logger.error(f"Error processing text content from {source.name}: {e}")

        return entries

    async def collect_all(self) -> Dict[str, List[Dict[str, Any]]]:
        """Collect commentaries from all configured sources."""
        all_commentaries = {}

        for source in self.sources:
            try:
                if source.type == "api":
                    entries = await self.collect_from_api(source)
                elif source.type == "web":
                    entries = await self.collect_from_web(source)
                else:  # local
                    entries = self.collect_from_local(source)

                if entries:
                    # Save raw entries
                    raw_path = os.path.join(self.raw_dir, f"{source.name.lower()}.json")
                    with open(raw_path, "w", encoding="utf-8") as f:
                        json.dump(entries, f, indent=2)

                    all_commentaries[source.name] = entries
                    logger.info(f"Collected {len(entries)} entries from {source.name}")

            except Exception as e:
                logger.error(f"Error collecting from {source.name}: {e}")

        return all_commentaries


async def main():
    """Run commentary collection."""
    collector = CommentaryCollector()
    commentaries = await collector.collect_all()
    print(f"Collected commentaries from {len(commentaries)} sources")


if __name__ == "__main__":
    asyncio.run(main())
