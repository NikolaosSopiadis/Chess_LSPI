from __future__ import annotations

from chess_rl.features.base import FeatureExtractor
from chess_rl.features.v1_basic import V1BasicFeatures
from chess_rl.features.v2_basic import V2BasicFeatures


_FEATURES: dict[str, type[FeatureExtractor]] = {
    "v1_basic": V1BasicFeatures,
    "v2_basic": V2BasicFeatures,
}


def get(name: str) -> FeatureExtractor:
    try:
        return _FEATURES[name]()
    except KeyError as e:
        raise ValueError(f"Unknown feature extractor: {name!r}") from e