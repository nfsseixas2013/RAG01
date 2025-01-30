import logging
from fastapi import APIRouter
from fastapi import HTTPException
from  app.config import settings
from app.models.output import health_output

logger = logging.getLogger(__name__)

router = APIRouter(prefix="")

@router.get("/health", response_model=health_output)
async def health():
    return{
        "status": f"I am a healthy {settings.api_slug}",
        "version": "1.0",
        "author": "Nilton Seixas"
    }
