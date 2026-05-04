from __future__ import annotations

from chess_rl.features.base import FeatureExtractor
from chess_rl.features.v1_basic import V1BasicFeatures
from chess_rl.features.v2_basic import V2BasicFeatures
from chess_rl.features.v2_1_basic import V21BasicFeatures
from chess_rl.features.v3_basic import V3BasicFeatures


_FEATURES: dict[str, type[FeatureExtractor]] = {
    "v1_basic": V1BasicFeatures,
    "v2_basic": V2BasicFeatures,
    "v2_1_basic": V21BasicFeatures,
    "v3_basic": V3BasicFeatures,
}


def get(name: str) -> FeatureExtractor:
    try:
        return _FEATURES[name]()
    except KeyError as e:
        raise ValueError(f"Unknown feature extractor: {name!r}") from e