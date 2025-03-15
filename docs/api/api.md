# Bible-AI API Documentation

This document provides comprehensive information about the Bible-AI API endpoints, parameters, and usage examples.

## API Overview

The Bible-AI API provides programmatic access to all core system capabilities, including Bible search, theological question answering, verse analysis, and more. The API is built using FastAPI and follows RESTful principles.

## Base URL

```
https://your-deployment-url.com/api/v1
```

For local development:

```
http://localhost:8000/api/v1
```

## Authentication

API requests require authentication using an API key:

```http
GET /api/v1/verses/search
Authorization: Bearer YOUR_API_KEY
```

To obtain an API key, contact the system administrator or generate one through the web interface if you have administrator privileges.

## Core Endpoints

### Bible Verse Lookup

Retrieve one or more Bible verses by reference.

```http
GET /api/v1/verses/{reference}
```

Parameters:
- `reference` (path): Bible verse reference (e.g., "John 3:16", "Romans 8:28-30")
- `translation` (query, optional): Bible translation to use (default: "NIV")

Example Response:
```json
{
  "reference": "John 3:16",
  "translation": "NIV",
  "text": "For God so loved the world that he gave his one and only Son, that whoever believes in him shall not perish but have eternal life.",
  "book": "John",
  "chapter": 3,
  "verse": 16
}
```

### Search Bible

Search the Bible using natural language or keywords.

```http
GET /api/v1/verses/search
```

Parameters:
- `query` (query): Search query
- `translation` (query, optional): Bible translation to use (default: "NIV")
- `limit` (query, optional): Maximum number of results (default: 10)

Example Response:
```json
{
  "results": [
    {
      "reference": "John 3:16",
      "text": "For God so loved the world...",
      "relevance": 0.95
    },
    {
      "reference": "1 John 4:9-10",
      "text": "This is how God showed his love among us...",
      "relevance": 0.87
    }
  ],
  "total_results": 127,
  "query": "God's love"
}
```

### Theological Q&A

Ask theological questions and receive answers with biblical support.

```http
POST /api/v1/theology/ask
```

Request Body:
```json
{
  "question": "What does the Bible teach about salvation?",
  "denominational_perspective": "reformed",
  "max_length": 500
}
```

Example Response:
```json
{
  "answer": "The Bible teaches that salvation is by grace through faith in Jesus Christ...",
  "supporting_verses": [
    {
      "reference": "Ephesians 2:8-9",
      "text": "For it is by grace you have been saved..."
    },
    {
      "reference": "Romans 10:9",
      "text": "If you declare with your mouth..."
    }
  ],
  "confidence": 0.92,
  "denominational_context": "reformed"
}
```

### Verse Contextual Analysis

Analyze a verse in its historical, cultural, and literary context.

```http
GET /api/v1/verses/{reference}/context
```

Parameters:
- `reference` (path): Bible verse reference
- `context_types` (query, optional): Types of context to analyze (historical, cultural, literary, canonical)

Example Response:
```json
{
  "reference": "1 Corinthians 11:14",
  "text": "Does not the very nature of things teach you that if a man has long hair, it is a disgrace to him?",
  "historical_context": "In first-century Corinth, men with long hair were often associated with...",
  "cultural_context": "Greek culture of the time had specific views about gender presentation...",
  "literary_context": "Paul is addressing issues of order and propriety in worship...",
  "canonical_context": "This passage relates to broader biblical themes of worship and gender..."
}
```

### Lexicon Word Study

Look up original Hebrew or Greek words and their meanings.

```http
GET /api/v1/lexicon/{word}
```

Parameters:
- `word` (path): Hebrew or Greek word (can be transliterated or in original script)
- `language` (query, optional): Specify "hebrew" or "greek" if known

Example Response:
```json
{
  "word": "ἀγάπη",
  "transliteration": "agape",
  "language": "greek",
  "definition": "love, goodwill, benevolence",
  "occurrences": 116,
  "key_verses": [
    {
      "reference": "1 Corinthians 13:13",
      "text": "And now these three remain: faith, hope and love. But the greatest of these is love."
    }
  ],
  "word_family": ["ἀγαπάω", "ἀγαπητός"],
  "semantic_domain": "Emotion, Attitude"
}
```

## Error Handling

The API uses standard HTTP status codes:

- `200 OK`: Request succeeded
- `400 Bad Request`: Invalid parameters
- `401 Unauthorized`: Authentication failed
- `404 Not Found`: Resource not found
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error

Error responses include details:

```json
{
  "error": {
    "code": "invalid_reference",
    "message": "Bible reference 'Genesis 51:1' is invalid. Genesis has only 50 chapters.",
    "details": {
      "book": "Genesis",
      "max_chapters": 50,
      "requested_chapter": 51
    }
  }
}
```

## Rate Limiting

API requests are subject to rate limiting:

- 60 requests per minute for authenticated users
- 5 requests per minute for unauthenticated requests

Rate limit information is included in response headers:

```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 59
X-RateLimit-Reset: 1620000000
```

## Webhooks

You can register webhooks to receive notifications:

```http
POST /api/v1/webhooks
```

Request Body:
```json
{
  "url": "https://your-service.com/webhook",
  "events": ["new_translation_available", "system_update"],
  "secret": "your_webhook_secret"
}
```

## SDKs and Client Libraries

Official client libraries are available for:

- Python: `pip install bible-ai-client`
- JavaScript: `npm install bible-ai-js`

Example Python usage:

```python
from bible_ai import BibleAI

client = BibleAI(api_key="YOUR_API_KEY")
verse = client.get_verse("John 3:16", translation="ESV")
print(verse.text)
```

## Support

For API support, please contact:
- Email: api-support@bible-ai.example.com
- Documentation issues: [GitHub Issues](https://github.com/yourusername/bible-ai/issues)