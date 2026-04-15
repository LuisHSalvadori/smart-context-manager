from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.endpoints import router

# Initialize FastAPI app with professional metadata
app = FastAPI(
    title="Smart Context Manager API",
    description="""
    API for intelligent PDF document analysis using vector embeddings and AI.
    
    **How to test:**
    1. Click the **Authorize** button and enter your `APP_SECURITY_TOKEN`.
    2. Use the `/upload-pdf` endpoint to process a document.
    3. Use the `/search` endpoint to ask questions based on the uploaded content.
    """,
    version="1.0.0"
)

# Configure CORS middleware using settings from config.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes from the endpoints module
app.include_router(router)