from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from dog_pose_mvp.api.exceptions import DogPoseApiError
from dog_pose_mvp.api.router import router
from dog_pose_mvp.api.service import _project_version


def _parse_cors_origins(raw_origins: str) -> list[str]:
    origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
    return origins or ["*"]


def _error_response(
    *,
    status_code: int,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    payload = {"error": {"code": code, "message": message, "details": details or {}}}
    return JSONResponse(status_code=status_code, content=payload)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Dog Pose FastAPI",
        version=_project_version(),
        description="FastAPI backend for dog pose inference and gait analysis.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_parse_cors_origins(os.getenv("DOG_POSE_CORS_ALLOW_ORIGINS", "*")),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(DogPoseApiError)
    async def handle_app_error(_: Request, exc: DogPoseApiError) -> JSONResponse:
        return _error_response(
            status_code=exc.status_code,
            code=exc.code,
            message=exc.message,
            details=exc.details,
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return _error_response(
            status_code=422,
            code="invalid_request",
            message="Request validation failed.",
            details={"errors": exc.errors()},
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(_: Request, exc: Exception) -> JSONResponse:
        return _error_response(
            status_code=500,
            code="internal_server_error",
            message="Unexpected server error.",
            details={"reason": str(exc)},
        )

    @app.get("/", tags=["meta"], summary="API metadata")
    async def root() -> dict[str, str]:
        return {
            "name": "Dog Pose FastAPI",
            "version": _project_version(),
            "docs": "/docs",
            "openapi": "/openapi.json",
        }

    app.include_router(router)
    return app


app = create_app()


def run() -> None:
    import uvicorn

    uvicorn.run(
        "dog_pose_mvp.api.main:app",
        host=os.getenv("DOG_POSE_API_HOST", "0.0.0.0"),
        port=int(os.getenv("DOG_POSE_API_PORT", "8000")),
        reload=False,
    )
