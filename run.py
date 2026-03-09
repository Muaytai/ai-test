"""Точка входа: сервер или обучение модели."""
import os
import sys
from pathlib import Path

# Загружаем .env из корня проекта (до любых импортов, использующих env)
_root = Path(__file__).resolve().parent
try:
    from dotenv import load_dotenv
    load_dotenv(_root / ".env")
except ImportError:
    pass

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        from app.train import main
        main()
    elif len(sys.argv) > 1 and sys.argv[1] == "download":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        from app.data import main
        main()
    else:
        import uvicorn
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", "8000"))
        uvicorn.run(
            "app.main:app",
            host=host,
            port=port,
            reload=True,
        )
