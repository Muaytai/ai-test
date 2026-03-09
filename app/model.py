"""Модель классификации кожных поражений (родинка vs меланома)."""
import os
import numpy as np
from PIL import Image

from .config import IMG_SIZE, MODEL_WEIGHTS_PATH, CLASS_NAMES

# Ленивая загрузка TensorFlow (тяжёлая зависимость)
_model = None


def _build_model():
    """Собирает модель: EfficientNetB0 (transfer learning) + бинарный классификатор."""
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    from tensorflow.keras.applications import EfficientNetB0

    base = EfficientNetB0(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
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


def get_model():
    """Возвращает синглтон модели, загружая веса если есть."""
    global _model
    if _model is None:
        _model = _build_model()
        if os.path.isfile(MODEL_WEIGHTS_PATH):
            _model.load_weights(MODEL_WEIGHTS_PATH)
    return _model


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Приводит изображение к формату модели: (1, H, W, 3), нормализация [0,1]."""
    img = image.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr[np.newaxis, ...]


def predict(image: Image.Image) -> dict:
    """
    Предсказание по одному изображению.
    Возвращает: class_name, confidence, is_risk, message.
    """
    model = get_model()
    x = preprocess_image(image)
    prob = float(model.predict(x, verbose=0)[0, 0])
    # prob — вероятность "malignant" (злокачественная)
    is_malignant = prob >= 0.5
    class_name = CLASS_NAMES[1] if is_malignant else CLASS_NAMES[0]
    confidence = prob if is_malignant else (1.0 - prob)

    if is_malignant:
        message = (
            "Модель оценивает образование как подозрительное (возможная меланома). "
            "Рекомендуется срочная консультация дерматолога или онколога."
        )
    else:
        message = (
            "Модель оценивает образование как доброкачественное. "
            "Это не заменяет осмотр врача — при любых изменениях родинки обратитесь к специалисту."
        )

    return {
        "class_name": class_name,
        "confidence": round(confidence, 4),
        "probability_malignant": round(prob, 4),
        "is_risk": is_malignant,
        "message": message,
    }
