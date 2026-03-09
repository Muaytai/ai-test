"""Конфигурация приложения."""
import os

# Размер входа модели (EfficientNet стандарт)
IMG_SIZE = 224
# Путь к весам модели (после обучения)
MODEL_WEIGHTS_PATH = os.path.join(
    os.path.dirname(__file__), "model_weights", "melanoma_model.weights.h5"
)
# Классы
CLASS_NAMES = ["benign", "malignant"]  # доброкачественная, злокачественная
