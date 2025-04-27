# API Endpoints

## Base URL
`http://localhost:8000`

## Authentication
None required (for now)

## Rate Limits
100 requests per minute per IP address

## Endpoints

### Health Check
`GET /health`

**Response:**
```json
{
  "status": "healthy"
}
```

### Single Prediction
`POST /predict`

**Request Body:**
```json
{
  "text": "John 3:16",
  "context": "The most famous verse of the Bible"
}
```

**Response:**
```json
{
  "text": "John 3:16",
  "prediction": 1,
  "confidence": 0.95,
  "theological_score": 0.85
}
```

### Batch Prediction
`POST /predict_batch`

**Request Body:**
```json
{
  "text": ["John 3:16", "Genesis 1:1"],
  "context": ["The most famous verse of the Bible", "Creation account"]
}
```

**Response:**
```json
{
  "results": [
    {
      "text": "John 3:16",
      "prediction": 1,
      "confidence": 0.95
    },
    {
      "text": "Genesis 1:1",
      "prediction": 2,
      "confidence": 0.98
    }
  ]
}
```

## Error Responses
- `400 Bad Request`: Invalid input
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error
