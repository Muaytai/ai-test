"""FastAPI приложение — API для анализа фото кожи."""
import io
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image

from .model import predict

# Корень проекта (родитель app/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "static"

app = FastAPI(
    title="AI Doctor — Анализ родинок и меланом",
    description="Загрузка фото кожного образования для предварительной оценки (не заменяет врача).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_image(file: UploadFile) -> Image.Image:
    data = file.file.read()
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Файл пустой")
    try:
        return Image.open(io.BytesIO(data)).copy()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Не удалось открыть изображение: {e}")


@app.post("/api/analyze")
async def analyze_skin(file: UploadFile = File(...)):
    """Принимает фото кожи, возвращает предсказание: доброкачественное / подозрительное (меланома)."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Нужен файл изображения (image/*)")
    img = _load_image(file)
    result = predict(img)
    return result


@app.get("/")
async def index():
    """Отдаёт главную страницу."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.is_file():
        raise HTTPException(status_code=500, detail="Static files not found")
    return FileResponse(index_path)


if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
