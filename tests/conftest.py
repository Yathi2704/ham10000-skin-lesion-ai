"""Shared fixtures."""

import numpy as np
import pytest
from PIL import Image

import data
from data import NUM_CLASSES


@pytest.fixture(scope="session")
def tiny_pool() -> data.Pool:
    """In-memory Pool: 10 random 32x32 images per class, 9 lesions per class (one lesion has 2 images).

    Needs no download and exercises lesion grouping.
    """
    rng = np.random.default_rng(0)
    images, ids, labels, lesions = {}, [], [], []
    for c in range(NUM_CLASSES):
        for k in range(10):
            image_id = f"FAKE_{c}_{k:02d}"
            images[image_id] = Image.fromarray(rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8))
            ids.append(image_id)
            labels.append(c)
            lesions.append(f"LES_{c}_{min(k, 8):02d}")  # images 08 and 09 share a lesion
    return data.Pool(
        image_ids=ids,
        labels=np.asarray(labels, dtype=np.int64),
        lesion_ids=np.asarray(lesions),
        images=data.MemoryImageStore(images),
    )
