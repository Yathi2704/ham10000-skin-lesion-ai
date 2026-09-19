"""Shared fixtures."""

import numpy as np
import pytest
from PIL import Image

from data import NUM_CLASSES


@pytest.fixture(scope="session")
def tiny_pool():
    """In-memory HF dataset: 10 random 32x32 images per class (no download needed)."""
    from datasets import Dataset, Features, Image as HFImage, Value

    rng = np.random.default_rng(0)
    images, labels, ids = [], [], []
    for c in range(NUM_CLASSES):
        for k in range(10):
            images.append(Image.fromarray(rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)))
            labels.append(c)
            ids.append(f"FAKE_{c}_{k:02d}")
    features = Features({"image": HFImage(), "label": Value("int64"), "image_id": Value("string")})
    return Dataset.from_dict({"image": images, "label": labels, "image_id": ids}, features=features)
