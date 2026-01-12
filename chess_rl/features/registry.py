from __future__ import annotations
from typing import Dict

from chess_rl.features.base import FeatureExtractor
from .v1_basic import V1BasicFeatures

FEATURES: Dict[str, FeatureExtractor] = {
    "v1_basic": V1BasicFeatures(),
}

def get(name: str) -> FeatureExtractor:
    return FEATURES[name]
