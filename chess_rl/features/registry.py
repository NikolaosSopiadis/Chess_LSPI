from __future__ import annotations

from chess_rl.features.base import FeatureExtractor
from chess_rl.features.v1_basic import V1BasicFeatures
from chess_rl.features.v2_basic import V2BasicFeatures
from chess_rl.features.v2_1_basic import V21BasicFeatures
from chess_rl.features.v3_basic import V3BasicFeatures
from chess_rl.features.v4_slim import V4SlimFeatures
from chess_rl.features.v5_center import V5CenterFeatures
from chess_rl.features.v6_attackmap import V6AttackMapFeatures
from chess_rl.features.v7_api_tactics import V7ApiTacticsFeatures
from chess_rl.features.v8_api_tactics_clean import V8ApiTacticsCleanFeatures
from chess_rl.features.v9_response_tactics import V9ResponseTacticsFeatures


_FEATURES: dict[str, type[FeatureExtractor]] = {
    "v1_basic":   V1BasicFeatures,
    "v2_basic":   V2BasicFeatures,
    "v2_1_basic": V21BasicFeatures,
    "v3_basic":   V3BasicFeatures,
    "v4_slim":    V4SlimFeatures,
    "v5_center":  V5CenterFeatures,
    "v6_attackmap": V6AttackMapFeatures,
    "v7_api_tactics": V7ApiTacticsFeatures,
    "v8_api_tactics_clean": V8ApiTacticsCleanFeatures,
    "v9_response_tactics": V9ResponseTacticsFeatures,
}


def get(name: str) -> FeatureExtractor:
    try:
        return _FEATURES[name]()
    except KeyError as e:
        raise ValueError(f"Unknown feature extractor: {name!r}") from e