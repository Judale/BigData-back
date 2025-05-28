from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw

# ----------------------------------------------------------------------
# Réseau identique à l'entraînement
# ----------------------------------------------------------------------
class BetterCNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.3),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.5),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

# ----------------------------------------------------------------------
# Conversion dessin QuickDraw -> image 28x28
# ----------------------------------------------------------------------
def strokes_to_canvas256(drawing: list, lw: int = 3, padding: int = 10) -> Image.Image:
    xs, ys = [], []
    for stroke in drawing:
        xs.extend(stroke[0]); ys.extend(stroke[1])
    if not xs:
        return Image.new("L", (256, 256), 0)

    xs, ys = np.array(xs), np.array(ys)
    min_x, max_x, min_y, max_y = xs.min(), xs.max(), ys.min(), ys.max()
    w, h = max_x - min_x, max_y - min_y
    w, h = max(w, 1e-3), max(h, 1e-3)

    avail = 256 - 2 * padding
    scale = avail / max(w, h)
    left = (256 - scale * w) / 2
    top  = (256 - scale * h) / 2

    img = Image.new("L", (256, 256), 0)
    drw = ImageDraw.Draw(img)
    for stroke in drawing:
        pts = [((x - min_x) * scale + left,
                (y - min_y) * scale + top) for x, y in zip(*stroke)]
        drw.line(pts, fill=255, width=lw)
    return img

def center_resize(img256: Image.Image, size: int = 28, pad: int = 2) -> np.ndarray:
    box = img256.getbbox()
    if box is None:
        return np.zeros((size, size), dtype=np.float32)
    crop = img256.crop(box)

    max_dim = size - 2 * pad
    w, h = crop.size
    if w >= h:
        new_w, new_h = max_dim, int(h * max_dim / w)
    else:
        new_h, new_w = max_dim, int(w * max_dim / h)
    crop = crop.resize((new_w, new_h), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (size, size), 0)
    canvas.paste(crop, ((size - new_w) // 2, (size - new_h) // 2))
    return np.asarray(canvas, dtype=np.float32)

# ----------------------------------------------------------------------
# Constantes
# ----------------------------------------------------------------------
CLASSES_10 = ["airplane", "angel", "apple", "axe", "banana", "bridge", "cup", "donut", "door", "mountain"]

CLASSES_51 = [
    "anvil", "apple", "axe", "backpack", "banana", "bed", "belt", "bicycle",
    "bird", "blueberry", "bread", "broccoli", "camel", "carrot", "cat",
    "cookie", "cow", "crab", "crocodile", "dolphin", "donut", "elephant",
    "eyeglasses", "fish", "flamingo", "flower", "fork", "grapes", "hamburger",
    "harp", "headphones", "hedgehog", "hexagon", "key", "knife", "ladder",
    "lion", "lollipop", "mushroom", "octopus", "pear", "pig", "pineapple",
    "pizza", "rabbit", "raccoon", "rhinoceros", "shark", "strawberry",
    "watermelon", "whale"
]

CLASSES_ANIMAUX = [
    "bird", "camel", "cat", "cow", "crab", "crocodile", "dolphin",
    "elephant", "fish", "flamingo", "hedgehog", "lion", "octopus",
    "pig", "rabbit", "raccoon", "rhinoceros", "shark", "whale"
]

CLASSES_OBJETS = [
    "alarm clock", "anvil", "axe", "backpack", "baseball bat", "bed",
    "belt", "bicycle", "cell phone", "eyeglasses", "flip flops", "flower",
    "fork", "harp", "headphones", "hexagon", "key", "knife", "ladder"
]

CLASSES_NOURRITURE = [
    "apple", "banana", "birthday_cake", "blueberry", "bread", "broccoli",
    "carrot", "cookie", "donut", "grapes", "hamburger", "hot dog",
    "ice_cream", "lollipop", "mushroom", "pear", "pineapple", "pizza",
    "strawberry", "watermelon"
]

MODEL_CONFIGS = {
    "default": {
        "weights": Path("backend/modele/prod/quickdraw_cnn.pth"),
        "classes": CLASSES_10,
    },
    "extended": {
        "weights": Path("backend/modele/prod/quickdraw_cnn_60_class.pth"),
        "classes":CLASSES_51,
    },
    "animaux": {
        "weights": Path("backend/modele/prod/Animaux-CNN.pth"),
        "classes": CLASSES_ANIMAUX,
    },
    "objets": {
        "weights": Path("backend/modele/prod/Objets-CNN.pth"),
        "classes": CLASSES_OBJETS,
    },
    "nourriture": {
        "weights": Path("backend/modele/prod/Nourritures.pth"),
        "classes": CLASSES_NOURRITURE,
    },
}

# ----------------------------------------------------------------------
# Chargement du modèle (une seule fois) + prédiction
# ----------------------------------------------------------------------
@lru_cache(maxsize=5)
def _get_model(model_type: Literal["default", "extended", "animaux", "objets", "nourriture"] = "default"):
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Type de modèle inconnu : {model_type}")

    config = MODEL_CONFIGS[model_type]
    weights = config["weights"]
    classes = config["classes"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BetterCNN(num_classes=len(classes)).to(device)

    if not weights.is_file():
        raise FileNotFoundError(f"Poids introuvables : {weights.resolve()}")
    state = torch.load(weights, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, device, classes

def predict(drawing: list, model_type: str = "default") -> tuple[str, float]:
    model, device, classes = _get_model(model_type)

    img256 = strokes_to_canvas256(drawing)
    img28 = center_resize(img256) / 255.0

    x = torch.from_numpy(img28[None, None]).to(device)
    with torch.no_grad():
        logits = model(x)
        idx = int(logits.argmax(1))
        prob = torch.softmax(logits, 1)[0, idx].item()

    return classes[idx], prob
