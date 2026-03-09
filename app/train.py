"""Обучение модели классификации меланомы. Вызывается через run.py train."""
import argparse
import os
import sys
import time

from pathlib import Path

# Корень проекта
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class _TrainingProgressCallback:
    """Печатает прогресс обучения после каждой эпохи: эпоха X/Y, метрики, ориентир по времени."""

    def __init__(self, total_epochs: int):
        self.total_epochs = total_epochs
        self.epoch_times: list[float] = []
        self._last_time: float = 0.0

    def on_train_begin(self, logs=None):
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            print("Обнаружена GPU:", gpus[0].name)
        else:
            print("Обучение на CPU (будет дольше)")
        print(f"Эпох: {self.total_epochs}. Прогресс по эпохам и метрики ниже.\n")
        self._last_time = time.perf_counter()

    def on_epoch_end(self, epoch: int, logs: dict):
        now = time.perf_counter()
        self.epoch_times.append(now - self._last_time)
        self._last_time = now
        epoch_1based = epoch + 1
        if self.epoch_times:
            avg_sec = sum(self.epoch_times) / len(self.epoch_times)
            remaining = self.total_epochs - epoch_1based
            eta_sec = avg_sec * remaining
            eta_str = f" | ориентир осталось: {int(eta_sec // 60)} мин"
        else:
            eta_str = ""
        loss = logs.get("loss", 0)
        auc = logs.get("auc", 0)
        val_loss = logs.get("val_loss", 0)
        val_auc = logs.get("val_auc", 0)
        print(
            f"Эпоха {epoch_1based}/{self.total_epochs} | "
            f"loss: {loss:.4f} | auc: {auc:.4f} | val_loss: {val_loss:.4f} | val_auc: {val_auc:.4f}{eta_str}"
        )
        sys.stdout.flush()


def _build_model(img_size: int):
    import tensorflow as tf
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras import layers, Model

    base = EfficientNetB0(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base.trainable = False
    x = layers.Dense(128, activation="relu")(base.output)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(1, activation="sigmoid", name="pred")(x)
    model = Model(base.input, out)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def run_train(
    data_dir: str | Path = None,
    epochs: int = 10,
    batch_size: int = 32,
    weights_dir: str | Path = None,
    img_size: int | None = None,
):
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    from .config import IMG_SIZE as DEFAULT_IMG_SIZE
    img_size = img_size or DEFAULT_IMG_SIZE

    data_dir = Path(data_dir or PROJECT_ROOT / "data")
    weights_dir = Path(weights_dir or PROJECT_ROOT / "app" / "model_weights")
    train_dir = data_dir
    benign = train_dir / "benign"
    malignant = train_dir / "malignant"

    if not benign.is_dir() or not malignant.is_dir():
        print("Нужны папки data/benign/ и data/malignant/ с изображениями.")
        print("Положите туда снимки и запустите снова: python run.py train")
        sys.exit(1)

    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.2,
    )
    train_ds = datagen.flow_from_directory(
        str(train_dir),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="binary",
        subset="training",
        classes=["benign", "malignant"],
    )
    val_ds = datagen.flow_from_directory(
        str(train_dir),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="binary",
        subset="validation",
        classes=["benign", "malignant"],
    )

    model = _build_model(img_size)
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights_path = weights_dir / "melanoma_model.weights.h5"

    progress_cb = _TrainingProgressCallback(total_epochs=epochs)
    callbacks_list = [
        tf.keras.callbacks.LambdaCallback(
            on_train_begin=progress_cb.on_train_begin,
            on_epoch_end=lambda e, l: progress_cb.on_epoch_end(e, l),
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(weights_path),
            save_best_only=True,
            monitor="val_auc",
            mode="max",
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            patience=3,
            mode="max",
            restore_best_weights=True,
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks_list,
    )
    model.save_weights(str(weights_path))
    print("Готово. Веса сохранены:", weights_path)


def main():
    parser = argparse.ArgumentParser(description="Обучение модели AI Doctor")
    parser.add_argument("--data-dir", default=None, help="Папка с data/benign и data/malignant (по умолчанию: data)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--weights-dir", default=None, help="Куда сохранить веса (по умолчанию: app/model_weights)")
    parser.add_argument("--img-size", type=int, default=None, help="Сторона изображения (по умолчанию 224). 192 — быстрее, чуть хуже качество. Если меняете: задайте тот же IMG_SIZE в app/config.py для инференса")
    args = parser.parse_args()
    run_train(
        data_dir=args.data_dir or os.path.join(PROJECT_ROOT, "data"),
        epochs=args.epochs,
        batch_size=args.batch_size,
        weights_dir=args.weights_dir,
        img_size=args.img_size,
    )


if __name__ == "__main__":
    main()
