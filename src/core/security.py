from fastapi import Security, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging
from src.core.config import settings

logger = logging.getLogger(__name__)

# Bearer Token Authentication
bearer_scheme = HTTPBearer(auto_error=False)

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(bearer_scheme)):
    """
    Verify Bearer token from Authorization header.

    Expects Authorization: Bearer <token> header with valid token.
    Can be disabled by setting REQUIRE_API_KEY=false in environment.

    Args:
        credentials: Bearer token credentials from Authorization header

    Returns:
        True if authentication is successful

    Raises:
        HTTPException: 401 if Bearer token is missing, 403 if invalid, 500 if misconfigured
    """
    # Allow bypass if API key requirement is disabled
    if not settings.REQUIRE_API_KEY:
        return True

    # Check if API key is configured
    if not settings.API_KEY:
        logger.warning("Bearer token not configured but REQUIRE_API_KEY is True. Denying access.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API authentication is not properly configured"
        )

    # Check if Bearer token is provided in request
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token is required. Provide Authorization: Bearer <token> header.",
            headers={"WWW-Authenticate": "Bearer"}
        )

    # Verify Bearer token matches configured API key
    if credentials.credentials != settings.API_KEY:
        logger.warning(f"Invalid bearer token attempt")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"}
        )

    return True
