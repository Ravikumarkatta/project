# src/serve/api.py
"""
FastAPI backend for Bible-AI.

Provides endpoints for scripture study, theological queries, and model inference with
security, monitoring, and theological validation.
"""

import json
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import jwt
from pathlib import Path
import asyncio
from datetime import datetime, timedelta
import logging
from src.utils.logger import get_logger
from src.theology.validator import TheologicalValidator
from src.theology.denominational import DenominationalAdjuster
from src.theology.controversial import ControversialHandler
from src.theology.pastoral import PastoralSensitivity
from src.model.architecture import BiblicalTransformer, BiblicalTransformerConfig
from src.monitoring.metrics import MetricsCollector
from src.serve.verse_resolver import VerseResolver
from src.serve.rate_limiter import RateLimiter
from src.serve.cache import Cache
from src.utils.security import verify_token, hash_password, verify_password

# Initialize logging
logger = get_logger("API")

# Load configurations
def load_config(config_path: str) -> Dict[str, Any]:
    """Load JSON configuration file with error handling."""
    try:
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with config_file.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {config_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to load config {config_path}: {e}")
        raise

security_config = load_config("config/security_config.json")
frontend_config = load_config("config/frontend_config.json")

# Initialize FastAPI app
app = FastAPI(
    title="Bible-AI API",
    description="API for intelligent scripture study and theological assistance",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=security_config["cors"]["allowed_origins"],
    allow_credentials=True,
    allow_methods=security_config["cors"]["allowed_methods"],
    allow_headers=security_config["cors"]["allowed_headers"],
)

# Initialize dependencies
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
rate_limiter = RateLimiter(
    requests_per_minute=security_config["rate_limiting"]["requests_per_minute"],
    burst_limit=security_config["rate_limiting"]["burst_limit"]
)
cache = Cache(ttl=frontend_config["api"]["cache_ttl"])
metrics_collector = MetricsCollector(port=8000)
validator = TheologicalValidator()
adjuster = DenominationalAdjuster()
handler = ControversialHandler()
sensitivity = PastoralSensitivity()
verse_resolver = VerseResolver()

# Load model
try:
    model_config = BiblicalTransformerConfig(**load_config("config/model_config.json"))
    model = BiblicalTransformer(model_config)
    model_path = Path("data/snapshots/best_model.pt")
    if model_path.exists():
        model.load_state_dict(torch.load(model_path))
    model.eval()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

# Pydantic models for request/response validation
class UserLogin(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)

class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=security_config["input_validation"]["max_input_length"])
    denomination: Optional[str] = "default"
    topic: Optional[str] = None

class VerseRequest(BaseModel):
    reference: str = Field(..., min_length=3, example="John 3:16")
    translation: str = Field(default="KJV")

class Token(BaseModel):
    access_token: str
    token_type: str

# Dependency for authenticated requests
async def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict[str, str]:
    """Verify JWT token and return user info."""
    try:
        payload = verify_token(token, security_config["authentication"]["secret_key"])
        return {"username": payload["sub"]}
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

# Authentication endpoint
@app.post("/token", response_model=Token)
async def login(user: UserLogin):
    """Authenticate user and issue JWT token."""
    # Mock user validation (replace with real DB check)
    stored_password_hash = hash_password("example_password")  # Placeholder
    if user.username != "admin" or not verify_password(user.password, stored_password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )

    access_token_expires = timedelta(seconds=security_config["authentication"]["token_expiry"])
    access_token = jwt.encode(
        {"sub": user.username, "exp": datetime.utcnow() + access_token_expires},
        security_config["authentication"]["secret_key"],
        algorithm="HS256"
    )
    logger.info(f"User {user.username} authenticated successfully")
    return {"access_token": access_token, "token_type": "bearer"}

# Health check endpoint
@app.get("/health")
async def health_check():
    """Check API health and system status."""
    start_time = datetime.now()
    status_dict = {
        "api": "healthy",
        "model": "loaded" if model is not None else "failed",
        "cache": "active" if cache.is_active() else "inactive",
        "metrics": "running" if metrics_collector.is_running() else "stopped"
    }
    latency = (datetime.now() - start_time).total_seconds()
    metrics_collector.track_inference(latency)
    logger.debug(f"Health check: {status_dict}")
    return status_dict

# Verse resolution endpoint
@app.get("/verse", response_model=Dict[str, str])
async def get_verse(
    request: VerseRequest,
    user: Dict[str, str] = Depends(get_current_user),
    client_ip: str = Depends(rate_limiter.limit)
):
    """Resolve and return Bible verse text."""
    cache_key = f"verse:{request.reference}:{request.translation}"
    cached_response = cache.get(cache_key)
    if cached_response:
        logger.debug(f"Cache hit for {cache_key}")
        return cached_response

    start_time = datetime.now()
    try:
        verse_text = verse_resolver.resolve(request.reference, request.translation)
        if not verse_text:
            raise HTTPException(status_code=404, detail="Verse not found")
        response = {"verse": verse_text}
        cache.set(cache_key, response)
        latency = (datetime.now() - start_time).total_seconds()
        metrics_collector.track_inference(latency)
        logger.info(f"Resolved verse {request.reference} in {request.translation}")
        return response
    except Exception as e:
        logger.error(f"Error resolving verse {request.reference}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Text generation endpoint
@app.post("/generate", response_model=Dict[str, Any])
async def generate_text(
    request: TextRequest,
    user: Dict[str, str] = Depends(get_current_user),
    client_ip: str = Depends(rate_limiter.limit)
):
    """Generate text with theological validation and adjustments."""
    cache_key = f"generate:{request.text}:{request.denomination}:{request.topic}"
    cached_response = cache.get(cache_key)
    if cached_response:
        logger.debug(f"Cache hit for {cache_key}")
        return cached_response

    start_time = datetime.now()
    try:
        # Tokenize input (assumes tokenizer in model)
        input_ids = model.tokenizer.tokenize(request.text)
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            predicted_text = model.tokenizer.detokenize(outputs["logits"].argmax(dim=-1))

        # Theological validation
        scores = validator.validate({"text": predicted_text})
        if scores["overall"] < validator.min_score:
            logger.warning(f"Low theological score for '{predicted_text}': {scores['overall']}")

        # Denominational adjustment
        adjusted = adjuster.adjust_for_denomination(predicted_text, request.denomination)
        text = adjusted["adjusted_text"]

        # Handle controversial topics
        controversy = handler.handle_controversy(text)
        text = controversy["adjusted_text"]

        # Apply pastoral sensitivity if topic provided
        if request.topic:
            sensitive = sensitivity.apply_sensitivity(text, request.topic)
            text = sensitive["adjusted_text"]

        response = {
            "text": text,
            "validation_scores": scores,
            "denomination_adjusted": adjusted["details"],
            "controversial": controversy["is_controversial"],
            "topics": [t["topic"] for t in controversy["topics"]]
        }
        cache.set(cache_key, response)
        latency = (datetime.now() - start_time).total_seconds()
        metrics_collector.track_inference(latency)
        metrics_collector.track_validation_score(scores)
        logger.info(f"Generated text for '{request.text[:50]}...' by {user['username']}")
        return response
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating text: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup."""
    logger.info("Starting Bible-AI API")
    metrics_collector.start()
    cache.start()
    if security_config["ssl"]["enabled"]:
        logger.info("SSL enabled; ensure certificates are configured")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    logger.info("Shutting down Bible-AI API")
    metrics_collector.stop()
    cache.stop()

if __name__ == "__main__":
    import uvicorn
    ssl_args = {
        "ssl_keyfile": security_config["ssl"]["key_path"],
        "ssl_certfile": security_config["ssl"]["cert_path"]
    } if security_config["ssl"]["enabled"] else {}
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        **ssl_args
    )