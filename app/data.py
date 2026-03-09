"""
Загрузка HAM10000 с Kaggle и раскладка по data/benign и data/malignant.
Запуск: python run.py download
Требуется: аккаунт Kaggle и API-ключ в ~/.kaggle/kaggle.json (или %USERPROFILE%\\.kaggle\\kaggle.json).
"""
import csv
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# Корень проекта
PROJECT_ROOT = Path(__file__).resolve().parent.parent
KAGGLE_DATASET = "kmader/skin-cancer-mnist-ham10000"
# dx -> malignant (True) или benign (False)
DX_MALIGNANT = {"mel", "bcc"}  # melanoma, basal cell carcinoma
DX_BENIGN = {"nv", "bkl", "akiec", "vasc", "df"}  # остальные в датасете


def _get_download_dir() -> Path:
    return PROJECT_ROOT / "data" / "downloads"


def _run_kaggle_download(dest: Path) -> bool:
    dest.mkdir(parents=True, exist_ok=True)
    try:
        # Без PIPE — вывод kaggle идёт в терминал, в память ничего не копим (меньше нагрузка на Cursor/ОЗУ)
        proc = subprocess.Popen(
            ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(dest)],
            stdout=None,
            stderr=None,
        )
        # Лёгкий прогресс: раз в 3 сек показываем размер скачанного (один zip, без тяжёлого glob)
        stop = threading.Event()
        approx_total_mb = 6000

        def show_progress():
            while not stop.is_set():
                try:
                    size = 0
                    for f in dest.iterdir():
                        if f.is_file() and f.suffix.lower() == ".zip":
                            size += f.stat().st_size
                            break
                    mb = size / (1024 * 1024)
                    pct = min(100, mb / approx_total_mb * 100) if approx_total_mb else 0
                    print(f"\rСкачано: {mb:.1f} / ~{approx_total_mb} MB ({pct:.0f}%)", end="", flush=True)
                except Exception:
                    pass
                stop.wait(3)
        t = threading.Thread(target=show_progress, daemon=True)
        t.start()
        try:
            proc.wait()
            if proc.returncode != 0:
                print("\nОшибка Kaggle (код выхода", proc.returncode + "). Проверьте токен и правила датасета.")
                return False
        finally:
            stop.set()
            t.join(timeout=4)
        print()
        return True
    except FileNotFoundError:
        print("Не найден kaggle. Установите: pip install kaggle")
        print("Настройте API: положите kaggle.json в ~/.kaggle/ (или .kaggle в папке пользователя).")
        return False
    except subprocess.CalledProcessError as e:
        print("Ошибка Kaggle:", e.stderr or e.stdout or str(e))
        print("Проверьте: 1) pip install kaggle  2) kaggle.json в ~/.kaggle/  3) приняты правила датасета на kaggle.com")
        return False


def _unzip_all(zip_path: Path, into: Path, show_progress: bool = True) -> None:
    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
        it = tqdm(names, desc="Распаковка", unit="файл") if (show_progress and tqdm) else names
        for name in it:
            z.extract(name, into)
    for child in into.iterdir():
        if child.suffix.lower() == ".zip":
            sub = into / child.stem
            sub.mkdir(exist_ok=True)
            _unzip_all(child, sub, show_progress=False)


def _find_metadata(root: Path) -> Path | None:
    for p in root.rglob("*.csv"):
        try:
            with open(p, newline="", encoding="utf-8", errors="ignore") as f:
                row = next(csv.DictReader(f))
                if "image_id" in row and "dx" in row:
                    return p
        except (StopIteration, KeyError):
            continue
    return None


def _find_image_by_id(root: Path, image_id: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        candidate = root / f"{image_id}{ext}"
        if candidate.is_file():
            return candidate
    for p in root.rglob(f"{image_id}.*"):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            return p
    return None


def _build_image_id_map(root: Path) -> dict[str, Path]:
    out = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            out[p.stem] = p
    return out


def run_download(
    data_dir: Path | None = None,
    download_dir: Path | None = None,
    limit_per_class: int | None = None,
) -> None:
    data_dir = data_dir or PROJECT_ROOT / "data"
    download_dir = download_dir or _get_download_dir()
    benign_dir = data_dir / "benign"
    malignant_dir = data_dir / "malignant"
    benign_dir.mkdir(parents=True, exist_ok=True)
    malignant_dir.mkdir(parents=True, exist_ok=True)

    # 1) Скачать архив
    print("Скачивание датасета с Kaggle...")
    if not _run_kaggle_download(download_dir):
        sys.exit(1)
    zips = list(download_dir.glob("*.zip"))
    if not zips:
        print("После скачивания не найден zip в", download_dir)
        sys.exit(1)
    main_zip = zips[0]
    extract_root = download_dir / "extracted"
    extract_root.mkdir(exist_ok=True)
    print("Распаковка...")
    _unzip_all(main_zip, extract_root)

    # 2) Найти метаданные
    meta_path = _find_metadata(extract_root)
    if not meta_path:
        print("В архиве не найден CSV с колонками image_id и dx.")
        sys.exit(1)
    print("Метаданные:", meta_path)

    # 3) Карта image_id -> путь к файлу
    image_map = _build_image_id_map(extract_root)
    print("Найдено изображений в архиве:", len(image_map))

    # 4) Читаем CSV и копируем
    with open(meta_path, newline="", encoding="utf-8", errors="ignore") as f:
        rows = [r for r in csv.DictReader(f) if r.get("image_id", "").strip() and (r.get("dx") or "").strip().lower()]
    benign_count = 0
    malignant_count = 0
    skipped_dx = set()
    missing = 0
    iter_rows = tqdm(rows, desc="Копирование в benign/malignant", unit="запись") if tqdm else rows
    for row in iter_rows:
        image_id = row.get("image_id", "").strip()
        dx = (row.get("dx", "") or "").strip().lower()
        if dx not in DX_MALIGNANT and dx not in DX_BENIGN:
            skipped_dx.add(dx)
            continue
        is_malignant = dx in DX_MALIGNANT
        if limit_per_class:
            if is_malignant and malignant_count >= limit_per_class:
                continue
            if not is_malignant and benign_count >= limit_per_class:
                continue
        path = image_map.get(image_id) or _find_image_by_id(extract_root, image_id)
        if not path or not path.is_file():
            missing += 1
            continue
        dest_dir = malignant_dir if is_malignant else benign_dir
        dest_file = dest_dir / f"{image_id}{path.suffix}"
        if not dest_file.exists() or path.stat().st_mtime > dest_file.stat().st_mtime:
            shutil.copy2(path, dest_file)
        if is_malignant:
            malignant_count += 1
        else:
            benign_count += 1

    if skipped_dx:
        print("Пропущены неизвестные dx:", skipped_dx)
    if missing:
        print("Не найдены файлы для записей:", missing)
    print("Готово. Доброкачественных:", benign_count, "| Злокачественных:", malignant_count)
    print("Папки:", benign_dir, "|", malignant_dir)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Скачать HAM10000 и разложить по data/benign и data/malignant")
    parser.add_argument("--data-dir", type=Path, default=None, help="Папка data (по умолчанию: data)")
    parser.add_argument("--limit", type=int, default=None, help="Макс. снимков на класс (для теста)")
    args = parser.parse_args()
    run_download(
        data_dir=args.data_dir,
        limit_per_class=args.limit,
    )


if __name__ == "__main__":
    main()
