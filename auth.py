"""
auth.py - Authentication and session management module
Handles password-based authentication and session management
"""

import hashlib
import secrets
import time
from typing import Optional, Dict
from datetime import datetime, timedelta
import logging
from fastapi import HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import Response
import json

logger = logging.getLogger(__name__)

class AuthManager:
    """
    Manages authentication sessions and password verification
    """

    def __init__(self, password: str, session_timeout_minutes: int = 480, cookie_secret: str = "default-secret"):
        """
        Initialize authentication manager

        Args:
            password: Plain text password (will be hashed)
            session_timeout_minutes: Session timeout in minutes (default 8 hours)
            cookie_secret: Secret key for signing session cookies
        """
        self.password_hash = self._hash_password(password)
        self.session_timeout_minutes = session_timeout_minutes
        self.cookie_secret = cookie_secret
        self.active_sessions: Dict[str, Dict] = {}

    def _hash_password(self, password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()

    def verify_password(self, password: str) -> bool:
        """
        Verify if provided password matches stored hash

        Args:
            password: Plain text password to verify

        Returns:
            True if password matches, False otherwise
        """
        password_hash = self._hash_password(password)
        logger.info(f"Verifying password: provided_hash={password_hash[:8]}..., stored_hash={self.password_hash[:8]}...")
        result = secrets.compare_digest(password_hash, self.password_hash)
        logger.info(f"Password verification result: {result}")
        return result

    def create_session(self) -> str:
        """
        Create a new authenticated session

        Returns:
            Session token
        """
        session_token = secrets.token_urlsafe(32)
        session_data = {
            "token": session_token,
            "created_at": time.time(),
            "expires_at": time.time() + (self.session_timeout_minutes * 60),
            "last_accessed": time.time()
        }

        self.active_sessions[session_token] = session_data
        logger.info(f"Created new session: {session_token[:8]}...")
        return session_token

    def validate_session(self, session_token: str) -> bool:
        """
        Validate if session token is still valid

        Args:
            session_token: Session token to validate

        Returns:
            True if valid, False otherwise
        """
        if not session_token or session_token not in self.active_sessions:
            return False

        session = self.active_sessions[session_token]
        current_time = time.time()

        # Check if session has expired
        if current_time > session["expires_at"]:
            self.remove_session(session_token)
            logger.info(f"Session expired: {session_token[:8]}...")
            return False

        # Update last accessed time
        session["last_accessed"] = current_time

        # Optionally extend session timeout
        session["expires_at"] = current_time + (self.session_timeout_minutes * 60)

        return True

    def remove_session(self, session_token: str) -> bool:
        """
        Remove a session (logout)

        Args:
            session_token: Session token to remove

        Returns:
            True if session was removed, False if not found
        """
        if session_token in self.active_sessions:
            del self.active_sessions[session_token]
            logger.info(f"Removed session: {session_token[:8]}...")
            return True
        return False

    def cleanup_expired_sessions(self) -> int:
        """
        Remove expired sessions

        Returns:
            Number of sessions removed
        """
        current_time = time.time()
        expired_tokens = [
            token for token, session in self.active_sessions.items()
            if current_time > session["expires_at"]
        ]

        for token in expired_tokens:
            del self.active_sessions[token]

        if expired_tokens:
            logger.info(f"Cleaned up {len(expired_tokens)} expired sessions")

        return len(expired_tokens)

    def get_session_count(self) -> int:
        """Get number of active sessions"""
        return len(self.active_sessions)

    def set_auth_cookie(self, response: Response, session_token: str) -> None:
        """
        Set authentication cookie in response

        Args:
            response: FastAPI response object
            session_token: Session token to store in cookie
        """
        expires = datetime.now() + timedelta(minutes=self.session_timeout_minutes)
        response.set_cookie(
            key="session_token",
            value=session_token,
            expires=expires,
            httponly=True,
            samesite="strict",
            secure=False  # Set to True if using HTTPS
        )

    def extract_token_from_request(self, request: Request) -> Optional[str]:
        """
        Extract session token from request (cookie or Authorization header)

        Args:
            request: FastAPI request object

        Returns:
            Session token if found, None otherwise
        """
        # First try cookie
        session_token = request.cookies.get("session_token")
        if session_token:
            return session_token

        # Then try Authorization header
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            return auth_header[7:]  # Remove "Bearer " prefix

        return None

# FastAPI dependency for protected routes
def get_current_user(auth_manager: AuthManager, request: Request) -> bool:
    """
    FastAPI dependency to check if user is authenticated

    Args:
        auth_manager: AuthManager instance
        request: FastAPI request object

    Returns:
        True if authenticated

    Raises:
        HTTPException: If not authenticated
    """
    session_token = auth_manager.extract_token_from_request(request)

    if not session_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated - no session token provided",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not auth_manager.validate_session(session_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired or invalid",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return True

def require_auth(auth_manager: AuthManager):
    """
    Decorator factory for requiring authentication on endpoints
    """
    def auth_dependency(request: Request):
        return get_current_user(auth_manager, request)

    return auth_dependency