"""
OSEDiff internal HTTP API (no auth). OpenAPI: /docs
"""
from __future__ import annotations

import io
import os
import sys
from contextlib import asynccontextmanager
from typing import Annotated, Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import Response
from PIL import Image

sys.path.append(os.getcwd())

from osediff_inference import EnhanceOptions, OSEDiffService, result_to_png_bytes

AlignMethodForm = Literal["adain", "wavelet"]

service = OSEDiffService()


def _parse_optional_px(raw: Optional[str], field: str) -> Optional[int]:
    """Vacío / omitido = None (usar upscale). Swagger envía string, no 0."""
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    try:
        value = int(text)
    except ValueError as exc:
        raise HTTPException(400, f"{field} debe ser un número entero; recibido: {raw!r}") from exc
    if value <= 0:
        return None
    if value < 8:
        raise HTTPException(400, f"{field} debe ser ≥ 8 px o dejarse vacío")
    return value


def _parse_align_method(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    text = raw.strip().lower()
    if not text:
        return None
    if text not in ("adain", "wavelet"):
        raise HTTPException(
            400,
            "align_method debe ser adain, wavelet o vacío (sin corrección)",
        )
    return text


def _parse_optional_upscale(raw: Optional[str]) -> Optional[int]:
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    try:
        value = int(text)
    except ValueError as exc:
        raise HTTPException(400, f"upscale debe ser un entero; recibido: {raw!r}") from exc
    if value < 1 or value > 8:
        raise HTTPException(400, "upscale debe estar entre 1 y 8 o dejarse vacío")
    return value


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    service.shutdown()


app = FastAPI(
    title="OSEDiff Image Enhance API",
    description=(
        "API interna para super-resolución / mejora de imagen con OSEDiff (general). "
        "Sin autenticación — usar solo en red privada."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    enhance = openapi_schema.get("paths", {}).get("/v1/enhance", {}).get("post")
    if enhance:
        content = enhance.get("requestBody", {}).get("content", {})
        form = content.get("multipart/form-data", {}).get("schema", {})
        props = form.get("properties", {})
        if "align_method" in props:
            props["align_method"]["enum"] = ["adain", "wavelet"]
            props["align_method"]["nullable"] = True
            props["align_method"].pop("default", None)
        required = form.setdefault("required", [])
        if "file" not in required:
            required.append("file")
    app.openapi_schema = openapi_schema
    return openapi_schema


app.openapi = custom_openapi


@app.get("/health", tags=["system"])
def health():
    if service.loading:
        status = "loading"
    elif service.ready:
        status = "ok"
    else:
        status = "idle"
    return {
        "status": status,
        "cuda": __import__("torch").cuda.is_available(),
        "model": "osediff" if service.ready else None,
        "model_loaded": service.ready,
    }


@app.post(
    "/v1/enhance",
    tags=["enhance"],
    summary="Mejorar imagen",
    response_class=Response,
    responses={
        200: {
            "content": {"image/png": {}},
            "description": "PNG mejorado. Headers: X-Prompt, X-Inference-Seconds, X-Width, X-Height",
        }
    },
)
def enhance_image(
    file: Annotated[UploadFile, File(description="Imagen (PNG, JPEG, WebP, etc.)")],
    width: Annotated[
        str,
        Form(
            description=(
                "Ancho objetivo (px). Vacío = no fijar ancho. "
                "Con height vacío, mantiene proporción. "
                "Vacío si usas upscale o height."
            ),
            json_schema_extra={"example": ""},
        ),
    ] = "",
    height: Annotated[
        str,
        Form(
            description=(
                "Altura objetivo (px). Vacío = no fijar altura. "
                "Con width vacío, mantiene proporción. "
                "Ambos = tamaño exacto (puede cambiar proporción)."
            ),
            json_schema_extra={"example": ""},
        ),
    ] = "",
    upscale: Annotated[
        str,
        Form(
            description="Factor de escala (1–8). Vacío si usas width/height.",
            json_schema_extra={"example": ""},
        ),
    ] = "",
    align_method: Annotated[
        Optional[AlignMethodForm],
        Form(description="adain o wavelet. Vacío = sin corrección."),
    ] = None,
    prompt: Annotated[
        str,
        Form(
            description=(
                "Texto extra al prompt RAM (etiquetas de maquillaje se filtran). "
                "Vacío añade: natural skin, realistic photo."
            ),
        ),
    ] = "",
):
    width_px = _parse_optional_px(width, "width")
    height_px = _parse_optional_px(height, "height")
    upscale_factor = _parse_optional_upscale(upscale)
    align = _parse_align_method(align_method)
    if width_px is None and height_px is None and upscale_factor is None:
        raise HTTPException(
            400,
            "Indica resolución objetivo (width y/o height) o upscale",
        )

    raw = file.file.read()
    if not raw:
        raise HTTPException(400, "Archivo vacío")

    try:
        image = Image.open(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(400, f"No se pudo leer la imagen: {exc}") from exc

    options = EnhanceOptions(
        upscale=upscale_factor,
        width=width_px,
        height=height_px,
        align_method=align,
        prompt=prompt,
    )

    try:
        result = service.enhance(image, options)
    except Exception as exc:
        raise HTTPException(500, f"Error en inferencia: {exc!r}") from exc

    prompt_header = result.prompt[:512].encode("ascii", "replace").decode("ascii")
    return Response(
        content=result_to_png_bytes(result),
        media_type="image/png",
        headers={
            "X-Prompt": prompt_header,
            "X-Inference-Seconds": f"{result.inference_seconds:.3f}",
            "X-Width": str(result.width),
            "X-Height": str(result.height),
        },
    )


def _uvicorn_log_config():
    """PM2 pinta stderr en rojo; uvicorn usa stderr para INFO por defecto."""
    import copy

    from uvicorn.config import LOGGING_CONFIG

    cfg = copy.deepcopy(LOGGING_CONFIG)
    cfg["handlers"]["default"]["stream"] = "ext://sys.stdout"
    cfg["loggers"]["uvicorn.error"] = {
        "handlers": ["default"],
        "level": "INFO",
        "propagate": False,
    }
    return cfg


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("OSEDIFF_HOST", "0.0.0.0")
    port = int(os.getenv("OSEDIFF_PORT", "8010"))
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=False,
        log_config=_uvicorn_log_config(),
        use_colors=False,
    )
