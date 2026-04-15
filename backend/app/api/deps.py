from fastapi import Header, HTTPException, status
from app.core.config import settings

async def verify_token(x_custom_token: str = Header(None)):
    """
    Dependency to validate the security token.
    Allows the Swagger UI to display the 'Authorize' field for testing.
    """
    if x_custom_token != settings.APP_SECURITY_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            # We use a clear message to help the developer/recruiter debug the error
            detail="Unauthorized: Invalid or missing Security Token"
        )
    return x_custom_token