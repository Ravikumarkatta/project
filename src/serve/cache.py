# src/serve/cache.py
"""
Caching system for Bible-AI API.

Provides a robust caching layer to store and retrieve API responses, improving performance.
Supports Redis for distributed caching with fallback to in-memory storage.
"""

import json
from typing import Any, Optional, Dict
from threading import Lock
import redis
from pathlib import Path
from datetime import datetime
from src.utils.logger import get_logger

logger = get_logger("Cache")


class Cache:
    """Manages caching of API responses with Redis or in-memory storage."""

    def __init__(self, ttl: int = 3600, redis_host: str = "localhost", redis_port: int = 6379, redis_db: int = 0) -> None:
        """
        Initialize the cache system.

        Args:
            ttl (int): Time-to-live for cache entries in seconds (default: 1 hour).
            redis_host (str): Redis server hostname.
            redis_port (int): Redis server port.
            redis_db (int): Redis database number.

        Raises:
            ValueError: If TTL is negative.
        """
        if ttl < 0:
            raise ValueError("TTL cannot be negative")
        self.ttl = ttl
        self.logger = logger
        self._lock = Lock()  # Thread-safety for in-memory fallback
        self._memory_cache: Dict[str, Dict[str, Any]] = {}  # In-memory fallback
        self._redis_client: Optional[redis.Redis] = None
        self._use_redis = False

        # Attempt to connect to Redis
        try:
            self._redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                decode_responses=True,
                socket_timeout=5
            )
            # Test connection
            self._redis_client.ping()
            self._use_redis = True
            self.logger.info(f"Connected to Redis at {redis_host}:{redis_port}, db={redis_db}")
        except redis.ConnectionError as e:
            self.logger.warning(f"Failed to connect to Redis: {e}. Falling back to in-memory cache.")
        except Exception as e:
            self.logger.error(f"Unexpected error initializing Redis: {e}. Using in-memory cache.")

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a value from the cache.

        Args:
            key (str): Cache key.

        Returns:
            Optional[Dict[str, Any]]: Cached value or None if not found/expired.

        Raises:
            ValueError: If key is empty or None.
        """
        if not key or not isinstance(key, str):
            self.logger.error(f"Invalid cache key: {key}")
            raise ValueError("Cache key must be a non-empty string")

        try:
            if self._use_redis and self._redis_client:
                value = self._redis_client.get(key)
                if value:
                    cached_data = json.loads(value)
                    self.logger.debug(f"Cache hit for key: {key}")
                    return cached_data
                self.logger.debug(f"Cache miss for key: {key}")
                return None
            else:
                with self._lock:
                    if key in self._memory_cache:
                        cached_data = self._memory_cache[key]
                        if cached_data["expiry"] > datetime.utcnow().timestamp():
                            self.logger.debug(f"Cache hit for key: {key}")
                            return cached_data["value"]
                        else:
                            del self._memory_cache[key]
                            self.logger.debug(f"Cache expired for key: {key}")
                    self.logger.debug(f"Cache miss for key: {key}")
                    return None
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to decode cached value for {key}: {e}")
            return None
        except redis.RedisError as e:
            self.logger.error(f"Redis error retrieving {key}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error retrieving {key}: {e}")
            return None

    def set(self, key: str, value: Dict[str, Any]) -> bool:
        """
        Store a value in the cache.

        Args:
            key (str): Cache key.
            value (Dict[str, Any]): Value to cache.

        Returns:
            bool: True if successfully cached, False otherwise.

        Raises:
            ValueError: If key is empty or value is not serializable.
        """
        if not key or not isinstance(key, str):
            self.logger.error(f"Invalid cache key: {key}")
            raise ValueError("Cache key must be a non-empty string")
        if not isinstance(value, dict):
            self.logger.error(f"Invalid cache value type: {type(value)}. Must be a dict.")
            raise ValueError("Cache value must be a dictionary")

        try:
            serialized_value = json.dumps(value)
        except TypeError as e:
            self.logger.error(f"Failed to serialize value for {key}: {e}")
            raise ValueError("Cache value must be JSON-serializable")

        try:
            if self._use_redis and self._redis_client:
                self._redis_client.setex(key, self.ttl, serialized_value)
                self.logger.debug(f"Cached {key} in Redis with TTL {self.ttl}s")
                return True
            else:
                with self._lock:
                    self._memory_cache[key] = {
                        "value": value,
                        "expiry": datetime.utcnow().timestamp() + self.ttl
                    }
                    self.logger.debug(f"Cached {key} in memory with TTL {self.ttl}s")
                    return True
        except redis.RedisError as e:
            self.logger.error(f"Redis error setting {key}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error setting {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """
        Remove a key from the cache.

        Args:
            key (str): Cache key to delete.

        Returns:
            bool: True if deleted, False if not found or error occurred.

        Raises:
            ValueError: If key is empty or None.
        """
        if not key or not isinstance(key, str):
            self.logger.error(f"Invalid cache key: {key}")
            raise ValueError("Cache key must be a non-empty string")

        try:
            if self._use_redis and self._redis_client:
                deleted = self._redis_client.delete(key)
                if deleted:
                    self.logger.debug(f"Deleted cache key: {key}")
                    return True
                self.logger.debug(f"Key {key} not found in Redis cache")
                return False
            else:
                with self._lock:
                    if key in self._memory_cache:
                        del self._memory_cache[key]
                        self.logger.debug(f"Deleted cache key: {key}")
                        return True
                    self.logger.debug(f"Key {key} not found in memory cache")
                    return False
        except redis.RedisError as e:
            self.logger.error(f"Redis error deleting {key}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error deleting {key}: {e}")
            return False

    def is_active(self) -> bool:
        """
        Check if the cache system is active.

        Returns:
            bool: True if Redis or in-memory cache is operational.
        """
        if self._use_redis and self._redis_client:
            try:
                self._redis_client.ping()
                return True
            except redis.RedisError:
                self.logger.warning("Redis cache is not responding")
                return False
        return True  # In-memory cache is always active if no Redis

    def start(self) -> None:
        """Start the cache system (placeholder for future async initialization)."""
        self.logger.info("Cache system started")

    def stop(self) -> None:
        """Stop the cache system and clean up resources."""
        if self._use_redis and self._redis_client:
            try:
                self._redis_client.close()
                self.logger.info("Redis connection closed")
            except redis.RedisError as e:
                self.logger.error(f"Error closing Redis connection: {e}")
        with self._lock:
            self._memory_cache.clear()
        self.logger.info("Cache system stopped")


if __name__ == "__main__":
    # Example usage
    cache = Cache(ttl=10)  # 10-second TTL for testing
    key = "test_key"
    value = {"data": "Hello, Bible-AI!"}

    # Set value
    assert cache.set(key, value), "Failed to set cache"
    print(f"Set {key}: {value}")

    # Get value
    retrieved = cache.get(key)
    print(f"Retrieved {key}: {retrieved}")
    assert retrieved == value, "Cache value mismatch"

    # Delete value
    assert cache.delete(key), "Failed to delete cache"
    assert cache.get(key) is None, "Cache not deleted"

    # Test invalid inputs
    try:
        cache.set("", value)
    except ValueError as e:
        print(f"Caught expected error: {e}")

    cache.stop()