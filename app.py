import json
import logging
import os
import time
from typing import Dict, List, Optional

import psutil
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from src.bible_manager.bible_handler import BibleHandler
from src.bible_manager.verse_reference import VerseReference, VerseReferenceDetector

# Setup logging
logging.basicConfig(
    filename="logs/api.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Bible AI API",
    description="API for Bible study and theological research",
    version="1.0.0",
)

DOWNLOAD_DIR = "downloads"
DATA_FILE = "data/processed/kjv_structured1.json"


# Load the Bible data
def load_bible_data() -> Dict:
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            logger.info(f"Successfully loaded Bible data from {DATA_FILE}")
            return data
    except Exception as e:
        logger.error(f"Error loading Bible data: {str(e)}")
        return {}


bible_data = load_bible_data()

# Initialize BibleHandler
bible_handler = BibleHandler()

# Initialize VerseReferenceDetector
verse_detector = VerseReferenceDetector()


class VerseResponse(BaseModel):
    book: str
    chapter: int
    verse: int
    text: str


class ChapterResponse(BaseModel):
    book: str
    chapter: int
    verses: Dict[str, str]


class SearchResult(BaseModel):
    book: str
    chapter: int
    verse: int
    text: str


class SearchResponse(BaseModel):
    query: str
    count: int
    results: List[SearchResult]


class CrossReferenceResponse(BaseModel):
    source_verse: VerseResponse
    cross_references: List[VerseResponse]
    relationship_type: str


class ContextAnalysisResponse(BaseModel):
    verse: VerseResponse
    historical_context: str
    literary_context: str
    theological_context: str
    chapter_context: str


class TextAnalysisRequest(BaseModel):
    text: str


class TextAnalysisResponse(BaseModel):
    references: List[VerseReference]
    formatted_references: List[str]


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime: float
    bible_data_loaded: bool


class MetricsResponse(BaseModel):
    cpu_usage: float
    memory_usage: float
    active_connections: int
    api_requests_total: int
    api_request_duration_seconds: Dict[str, float]


start_time = time.time()
request_count = 0
request_durations: Dict[str, List[float]] = {}


@app.get("/")
async def home():
    """
    Root endpoint to confirm the API is running.
    """
    return {
        "message": "Welcome to Bible AI backend service.",
        "endpoints": {
            "/api/v1/verse/{book}/{chapter}/{verse}": "Get a specific verse",
            "/api/v1/chapter/{book}/{chapter}": "Get an entire chapter",
            "/api/v1/search": "Search the Bible text",
            "/api/v1/books": "List all available books",
            "/api/v1/cross-references/{book}/{chapter}/{verse}": "Find cross-references for a specific verse",
            "/api/v1/context/{book}/{chapter}/{verse}": "Get contextual analysis for a specific verse",
            "/api/v1/bible/upload": "Upload a new Bible version",
            "/api/v1/bible/download/{version}": "Download a specific Bible version",
            "/api/v1/bible/versions": "List all available Bible versions",
            "/api/v1/analyze/references": "Analyze text to detect Bible verse references",
            "/health": "Health check endpoint",
            "/metrics": "Metrics endpoint",
        },
    }


@app.post("/api/v1/bible/upload")
async def upload_bible(file: UploadFile = File(...), version: str = None):
    """
    Upload a new Bible version.
    """
    if not version:
        raise HTTPException(status_code=400, detail="Version name is required")

    content = await file.read()
    file_content = content.decode("utf-8")
    return await bible_handler.process_uploaded_bible(file_content, version)


@app.get("/api/v1/bible/download/{version}")
async def download_bible(version: str):
    """
    Download a specific Bible version.
    """
    return await bible_handler.download_bible_version(version)


@app.get("/api/v1/bible/versions")
async def list_versions():
    """
    List all available Bible versions.
    """
    return await bible_handler.list_available_versions()


@app.get("/api/v1/verse/{book}/{chapter}/{verse}", response_model=VerseResponse)
async def get_verse(book: str, chapter: int, verse: int):
    """
    Retrieve a specific verse from the Bible.
    """
    if book in bible_data:
        chapter_data = bible_data[book].get(str(chapter))
        if chapter_data:
            verse_text = chapter_data.get(str(verse))
            if verse_text:
                logger.info(f"Retrieved verse: {book} {chapter}:{verse}")
                return {
                    "book": book,
                    "chapter": chapter,
                    "verse": verse,
                    "text": verse_text,
                }
    logger.error(f"Verse not found: {book} {chapter}:{verse}")
    raise HTTPException(status_code=404, detail="Verse not found")


@app.get("/api/v1/chapter/{book}/{chapter}", response_model=ChapterResponse)
async def get_chapter(book: str, chapter: int):
    """
    Retrieve an entire chapter from the Bible.
    """
    if book in bible_data:
        chapter_data = bible_data[book].get(str(chapter))
        if chapter_data:
            logger.info(f"Retrieved chapter: {book} {chapter}")
            return {"book": book, "chapter": chapter, "verses": chapter_data}
    logger.error(f"Chapter not found: {book} {chapter}")
    raise HTTPException(status_code=404, detail="Chapter not found")


@app.get("/api/v1/search", response_model=SearchResponse)
async def search_bible(q: str, book: Optional[str] = None):
    """
    Search the Bible text.
    """
    if not q or q.strip() == "":
        raise HTTPException(status_code=400, detail="Search query cannot be empty")

    results = []
    try:
        books_to_search = [book] if book and book in bible_data else bible_data.keys()
        query = q.lower()

        for book_name in books_to_search:
            book_data = bible_data[book_name]
            for chapter in book_data:
                for verse, text in book_data[chapter].items():
                    if query in text.lower():
                        results.append(
                            {
                                "book": book_name,
                                "chapter": int(chapter),
                                "verse": int(verse),
                                "text": text,
                            }
                        )

        logger.info(f"Search query '{q}' returned {len(results)} results")
        return {
            "query": q,
            "count": len(results),
            "results": results[:50],  # Limit results to 50 matches
        }
    except Exception as e:
        logger.error(f"Error during search: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/books")
async def list_books():
    """
    List all available books in the Bible.
    """
    try:
        books = list(bible_data.keys())
        logger.info(f"Retrieved list of {len(books)} books")
        return {"count": len(books), "books": books}
    except Exception as e:
        logger.error(f"Error listing books: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/api/v1/cross-references/{book}/{chapter}/{verse}",
    response_model=CrossReferenceResponse,
)
async def get_cross_references(book: str, chapter: int, verse: int):
    """
    Find cross-references for a specific verse.
    """
    # First get the source verse
    try:
        source_verse = await get_verse(book, chapter, verse)
    except HTTPException as e:
        if e.status_code == 404:
            logger.error(
                f"Source verse not found for cross-references: {book} {chapter}:{verse}"
            )
            raise HTTPException(status_code=404, detail="Source verse not found")
        raise

    # Example cross-reference logic (to be enhanced with actual cross-reference database)
    # For now, return related verses based on simple text matching
    search_terms = source_verse["text"].lower().split()
    cross_refs = []

    for b in bible_data:
        for ch in bible_data[b]:
            for v, text in bible_data[b][ch].items():
                if b == book and int(ch) == chapter and int(v) == verse:
                    continue

                # Simple matching logic - look for verses with similar key terms
                matches = sum(
                    1 for term in search_terms if term.lower() in text.lower()
                )
                if matches >= 2:  # At least 2 matching terms
                    cross_refs.append(
                        {"book": b, "chapter": int(ch), "verse": int(v), "text": text}
                    )

    logger.info(
        f"Found {len(cross_refs)} cross-references for {book} {chapter}:{verse}"
    )
    return {
        "source_verse": source_verse,
        "cross_references": cross_refs[:5],  # Limit to top 5 matches
        "relationship_type": "thematic",  # To be enhanced with actual relationship classification
    }


@app.get(
    "/api/v1/context/{book}/{chapter}/{verse}", response_model=ContextAnalysisResponse
)
async def get_verse_context(book: str, chapter: int, verse: int):
    """
    Get contextual analysis for a specific verse.
    """
    # Get the target verse
    try:
        verse_data = await get_verse(book, chapter, verse)
    except HTTPException as e:
        if e.status_code == 404:
            logger.error(
                f"Source verse not found for context analysis: {book} {chapter}:{verse}"
            )
            raise HTTPException(status_code=404, detail="Source verse not found")
        raise

    # Get surrounding verses for chapter context
    try:
        chapter_data = await get_chapter(book, chapter)
        verses = chapter_data["verses"]
        verse_numbers = sorted([int(v) for v in verses.keys()])
        current_idx = verse_numbers.index(verse)

        # Get 2 verses before and after for immediate context
        start_idx = max(0, current_idx - 2)
        end_idx = min(len(verse_numbers), current_idx + 3)
        context_verses = [
            f"v{verse_numbers[i]}: {verses[str(verse_numbers[i])]}"
            for i in range(start_idx, end_idx)
        ]

        logger.info(f"Generated context analysis for {book} {chapter}:{verse}")
        return {
            "verse": verse_data,
            "historical_context": "Historical context to be enhanced with actual historical data",
            "literary_context": "Literary context to be enhanced with genre and literary analysis",
            "theological_context": "Theological context to be enhanced with theological analysis",
            "chapter_context": "\n".join(context_verses),
        }
    except HTTPException as e:
        if e.status_code == 404:
            logger.error(f"Chapter not found for context analysis: {book} {chapter}")
            raise HTTPException(status_code=404, detail="Chapter not found")
        raise


@app.post("/api/v1/analyze/references", response_model=TextAnalysisResponse)
async def analyze_verse_references(request: TextAnalysisRequest):
    """
    Analyze text to detect Bible verse references.
    """
    try:
        references = verse_detector.detect_references(request.text)
        formatted_refs = [verse_detector.format_reference(ref) for ref in references]

        logger.info(f"Found {len(references)} verse references in text")
        return {"references": references, "formatted_references": formatted_refs}
    except Exception as e:
        logger.error(f"Error analyzing verse references: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint for monitoring.
    """
    return {
        "status": "healthy",
        "version": "1.0.0",
        "uptime": time.time() - start_time,
        "bible_data_loaded": bool(bible_data),
    }


@app.get("/metrics", response_model=MetricsResponse)
async def metrics():
    """
    Metrics endpoint for monitoring system performance.
    """
    cpu_percent = psutil.cpu_percent()
    memory = psutil.Process().memory_info()
    memory_percent = memory.rss / psutil.virtual_memory().total * 100

    # Calculate average request durations
    avg_durations = {
        endpoint: sum(durations) / len(durations) if durations else 0
        for endpoint, durations in request_durations.items()
    }

    return {
        "cpu_usage": cpu_percent,
        "memory_usage": memory_percent,
        "active_connections": len(psutil.Process().connections()),
        "api_requests_total": request_count,
        "api_request_duration_seconds": avg_durations,
    }


@app.middleware("http")
async def track_metrics(request, call_next):
    global request_count
    request_count += 1

    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    # Track request duration by endpoint
    endpoint = request.url.path
    if endpoint not in request_durations:
        request_durations[endpoint] = []
    request_durations[endpoint].append(duration)

    # Keep only last 1000 requests per endpoint
    if len(request_durations[endpoint]) > 1000:
        request_durations[endpoint] = request_durations[endpoint][-1000:]

    return response


if __name__ == "__main__":
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)
