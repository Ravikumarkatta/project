"""
Security utilities for Bible-AI application.

This module provides security-related functionality including
input sanitization, authentication helpers, authorization checks,
and protection against common web vulnerabilities.
"""

import re
import hashlib
import secrets
import json
import os
import time
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import base64
import hmac
from datetime import datetime, timedelta

# Regular expressions for security validation
SAFE_STRING_PATTERN = re.compile(r'^[\w\-\.\:\s\,\;\"\'\(\)\[\]\{\}\?\!]*$')
BIBLE_REFERENCE_PATTERN = re.compile(r'^[\w\s\d\:\-\.]+$')  # Allows Bible references like "John 3:16-17"
EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

# Default security settings
DEFAULT_PASSWORD_MIN_LENGTH = 10
DEFAULT_TOKEN_EXPIRY = 24 * 60 * 60  # 24 hours in seconds
DEFAULT_MAX_LOGIN_ATTEMPTS = 5
DEFAULT_LOCKOUT_TIME = 15 * 60  # 15 minutes in seconds

class SecurityManager:
    """Manager for application security features."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize SecurityManager with optional configuration.
        
        Args:
            config_path: Path to security configuration JSON file
        """
        # Default settings
        self.config = {
            'password_min_length': DEFAULT_PASSWORD_MIN_LENGTH,
            'token_expiry': DEFAULT_TOKEN_EXPIRY,
            'max_login_attempts': DEFAULT_MAX_LOGIN_ATTEMPTS,
            'lockout_time': DEFAULT_LOCKOUT_TIME,
            'allowed_origins': ['http://localhost:3000'],
            'csrf_protection': True,
            'rate_limiting': True
        }
        
        # Load configuration if provided
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    self.config.update(loaded_config)
            except Exception as e:
                print(f"Error loading security config: {e}")
        
        # Secret key for token signing
        self.secret_key = os.environ.get('BIBLE_AI_SECRET_KEY')
        if not self.secret_key:
            # Generate a secret key if not provided (not recommended for production)
            self.secret_key = secrets.token_hex(32)
            print("Warning: Using generated secret key. Set BIBLE_AI_SECRET_KEY environment variable for production.")
        
        # Failed login tracking
        self.failed_logins = {}  # username/IP -> {count: int, last_attempt: timestamp}
    
    def sanitize_input(self, text: str, pattern: Optional[re.Pattern] = None) -> Tuple[str, bool]:
        """
        Sanitize user input to prevent injection attacks.
        
        Args:
            text: Input text to sanitize
            pattern: Optional regex pattern to validate against (uses SAFE_STRING_PATTERN by default)
            
        Returns:
            Tuple of (sanitized_text, is_safe)
        """
        if text is None:
            return ("", False)
            
        # Remove potentially dangerous characters
        sanitized = text.strip()
        
        # Check against pattern
        validation_pattern = pattern or SAFE_STRING_PATTERN
        is_safe = bool(validation_pattern.match(sanitized))
        
        return (sanitized, is_safe)
    
    def sanitize_bible_reference(self, reference: str) -> Tuple[str, bool]:
        """
        Sanitize a Bible reference input.
        
        Args:
            reference: Bible reference string (e.g., "John 3:16")
            
        Returns:
            Tuple of (sanitized_reference, is_safe)
        """
        return self.sanitize_input(reference, BIBLE_REFERENCE_PATTERN)
    
    def hash_password(self, password: str, salt: Optional[str] = None) -> Tuple[str, str]:
        """
        Securely hash a password using PBKDF2.
        
        Args:
            password: Plain text password
            salt: Optional salt (generates a new one if not provided)
            
        Returns:
            Tuple of (password_hash, salt)
        """
        if salt is None:
            salt = secrets.token_hex(16)
            
        # Hash password with salt using PBKDF2
        key = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000  # Number of iterations
        )
        
        # Convert to hex
        password_hash = key.hex()
        
        return (password_hash, salt)
    
    def verify_password(self, stored_hash: str, stored_salt: str, provided_password: str) -> bool:
        """
        Verify a password against a stored hash.
        
        Args:
            stored_hash: Previously hashed password
            stored_salt: Salt used for hashing
            provided_password: Password to verify
            
        Returns:
            True if password matches, False otherwise
        """
        generated_hash, _ = self.hash_password(provided_password, stored_salt)
        return secrets.compare_digest(generated_hash, stored_hash)
    
    def validate_password_strength(self, password: str) -> Tuple[bool, str]:
        """
        Check if a password meets security requirements.
        
        Args:
            password: Password to validate
            
        Returns:
            Tuple of (is_valid, reason)
        """
        if len(password) < self.config['password_min_length']:
            return (False, f"Password must be at least {self.config['password_min_length']} characters")
            
        # Check for character variety
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(not c.isalnum() for c in password)
        
        if not (has_upper and has_lower and has_digit and has_special):
            return (False, "Password must include uppercase, lowercase, number, and special characters")
            
        return (True, "Password meets requirements")
    
    def generate_token(self, user_id: str, expiry_seconds: Optional[int] = None) -> str:
        """
        Generate a secure authentication token.
        
        Args:
            user_id: User identifier to include in token
            expiry_seconds: Token validity duration in seconds
            
        Returns:
            Secure token string
        """
        if expiry_seconds is None:
            expiry_seconds = self.config['token_expiry']
            
        # Create token payload
        expiry = int(time.time() + expiry_seconds)
        payload = {
            'user_id': user_id,
            'exp': expiry,
            'iat': int(time.time()),
            'jti': secrets.token_hex(8)  # Token ID for potential revocation
        }
        
        # Encode payload
        payload_bytes = json.dumps(payload).encode('utf-8')
        payload_b64 = base64.urlsafe_b64encode(payload_bytes).decode('utf-8').rstrip('=')
        
        # Create signature
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            payload_b64.encode('utf-8'),
            hashlib.sha256
        ).digest()
        signature_b64 = base64.urlsafe_b64encode(signature).decode('utf-8').rstrip('=')
        
        # Combine payload and signature
        token = f"{payload_b64}.{signature_b64}"
        
        return token
    
    def verify_token(self, token: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify a token's authenticity and extract payload.
        
        Args:
            token: Token to verify
            
        Returns:
            Tuple of (is_valid, payload)
        """
        try:
            # Split token
            if '.' not in token:
                return (False, {})
                
            payload_b64, signature_b64 = token.split('.')
            
            # Add padding if needed
            payload_b64 += '=' * (-len(payload_b64) % 4)
            signature_b64 += '=' * (-len(signature_b64) % 4)
            
            # Decode payload
            payload_bytes = base64.urlsafe_b64decode(payload_b64)
            payload = json.loads(payload_bytes)
            
            # Verify signature
            expected_signature = hmac.new(
                self.secret_key.encode('utf-8'),
                payload_b64.rstrip('=').encode('utf-8'),
                hashlib.sha256
            ).digest()
            actual_signature = base64.urlsafe_b64decode(signature_b64)
            
            if not secrets.compare_digest(expected_signature, actual_signature):
                return (False, {})
                
            # Check expiration
            if payload.get('exp', 0) < time.time():
                return (False, {})
                
            return (True, payload)
        except Exception:
            return (False, {})
    
    def track_login_attempt(self, identifier: str, success: bool) -> Tuple[bool, int]:
        """
        Track login attempts to prevent brute force attacks.
        
        Args:
            identifier: Username or IP address
            success: Whether login was successful
            
        Returns:
            Tuple of (is_locked_out, attempts_remaining)
        """
        current_time = time.time()
        
        # Reset tracking on successful login
        if success:
            if identifier in self.failed_logins:
                del self.failed_logins[identifier]
            return (False, self.config['max_login_attempts'])
        
        # Get or initialize tracking info
        tracking_info = self.failed_logins.get(identifier, {'count': 0, 'last_attempt': 0})
        
        # Check lockout expiration
        lockout_expiry = tracking_info['last_attempt'] + self.config['lockout_time']
        if tracking_info['count'] >= self.config['max_login_attempts'] and current_time < lockout_expiry:
            # Still locked out
            time_remaining = int(lockout_expiry - current_time)
            return (True, time_remaining)
        elif tracking_info['count'] >= self.config['max_login_attempts']:
            # Lockout expired, reset counter
            tracking_info = {'count': 1, 'last_attempt': current_time}
        else:
            # Increment count
            tracking_info['count'] += 1
            tracking_info['last_attempt'] = current_time
        
        # Update tracking
        self.failed_logins[identifier] = tracking_info
        attempts_remaining = max(0, self.config['max_login_attempts'] - tracking_info['count'])
        
        return (False, attempts_remaining)
    
    def generate_csrf_token(self) -> str:
        """
        Generate a CSRF token to prevent cross-site request forgery.
        
        Returns:
            CSRF token string
        """
        return secrets.token_hex(16)
    
    def is_allowed_origin(self, origin: str) -> bool:
        """
        Check if an origin is allowed for CORS.
        
        Args:
            origin: Origin header value
            
        Returns:
            True if origin is allowed
        """
        return origin in self.config['allowed_origins']
    
    def sanitize_theological_text(self, text: str) -> str:
        """
        Sanitize theological text input while preserving special characters
        commonly found in Bible verses and theological texts.
        
        Args:
            text: Theological text to sanitize
            
        Returns:
            Sanitized text
        """
        # Remove potentially malicious HTML/script tags
        # but keep theological special characters
        sanitized = re.sub(r'<[^>]*>', '', text)
        return sanitized.strip()
    
    def validate_api_request(self, 
                            endpoint: str, 
                            method: str, 
                            user_role: str,
                            required_role: str) -> bool:
        """
        Validate if a user has permission to access an API endpoint.
        
        Args:
            endpoint: API endpoint path
            method: HTTP method (GET, POST, etc.)
            user_role: Role of the requesting user
            required_role: Role required for the endpoint
            
        Returns:
            True if access is allowed
        """
        # Simple role hierarchy: admin > editor > user > guest
        role_levels = {
            'admin': 4,
            'editor': 3,
            'user': 2,
            'guest': 1,
            '': 0
        }
        
        user_level = role_levels.get(user_role.lower(), 0)
        required_level = role_levels.get(required_role.lower(), 0)
        
        return user_level >= required_level


# Create a default security manager instance
default_security_manager = SecurityManager()

# Convenience functions using the default manager
sanitize_input = default_security_manager.sanitize_input
hash_password = default_security_manager.hash_password
verify_password = default_security_manager.verify_password
generate_token = default_security_manager.generate_token
verify_token = default_security_manager.verify_token
sanitize_bible_reference = default_security_manager.sanitize_bible_reference
sanitize_theological_text = default_security_manager.sanitize_theological_text
