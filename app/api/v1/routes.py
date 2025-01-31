import logging
from fastapi import APIRouter
from fastapi import HTTPException
from  app.config import settings
from app.models.output import health_output
from app.api.v1.rag import rag_api

logger = logging.getLogger(__name__)

router = APIRouter(prefix="")

@router.get("/health", response_model=health_output)
async def health():
    return{
        "status": f"I am a healthy {settings.api_slug}",
        "version": "1.0",
        "author": "Nilton Seixas"
    }

router.include_router(rag_api.app)
