from fastapi import FastAPI

from app.api.exception_handlers import register_exception_handlers
from app.api.middleware import request_context_middleware
from app.api.v1.router import api_router
from app.core.config import settings
from app.core.logging import configure_logging
from app.core.runtime import api_lifespan, scheduler_lifespan, worker_lifespan


def create_api_app() -> FastAPI:
    configure_logging()
    application = FastAPI(title=settings.app_name, version="0.1.0", lifespan=api_lifespan)
    application.middleware("http")(request_context_middleware)
    register_exception_handlers(application)
    application.include_router(api_router, prefix="/api/v1")
    return application


def create_scheduler_app() -> FastAPI:
    configure_logging()
    application = FastAPI(
        title=f"{settings.app_name}-scheduler",
        version="0.1.0",
        lifespan=scheduler_lifespan,
    )
    application.middleware("http")(request_context_middleware)
    register_exception_handlers(application)
    application.include_router(api_router, prefix="/api/v1")
    return application


def create_worker_app() -> FastAPI:
    configure_logging()
    application = FastAPI(
        title=f"{settings.app_name}-worker",
        version="0.1.0",
        lifespan=worker_lifespan,
    )
    application.middleware("http")(request_context_middleware)
    register_exception_handlers(application)
    application.include_router(api_router, prefix="/api/v1")
    return application


app = create_api_app()
scheduler_app = create_scheduler_app()
worker_app = create_worker_app()
