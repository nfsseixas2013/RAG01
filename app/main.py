import logging
from starlette.middleware.cors import CORSMiddleware as CORSMiddleware
from app.api.v1 import routes
from app.config import settings
from fastapi import FastAPI
from mangum import Mangum

logger = logging.getLogger(__name__)

app = FastAPI(
    docs_url=settings.base_path + "/docs",
#    redoc_url=settings.base_path + "/redocs",
    title=settings.api_slug,
#    description=settings.description,
#    version=settings.version,
    debug=True,
)
origins = ["*",]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    routes.router,
    prefix=settings.base_path,
    responses={404: {"description": "Not found"}},
)

handler = Mangum(app)