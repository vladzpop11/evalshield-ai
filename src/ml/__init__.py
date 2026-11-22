"""EvalShield-AI Machine Learning Module."""

from .champion_challenger import ChampionChallengerRouter, get_champion_challenger_router
from .drift_recalibration import DriftRecalibrationEngine, RecalibrationResult, get_recalibration_engine
from .feature_extractor import TextFeatureExtractor, get_feature_extractor
from .inference import EvalShieldEngine, get_inference_engine
from .model_generator import generate_shield_model
from .pii_detector import PIIDetector, PIIEntity, get_pii_detector
from .unicode_sanitizer import UnicodeSanitizer, get_unicode_sanitizer

__all__ = [
    "EvalShieldEngine",
    "get_inference_engine",
    "generate_shield_model",
    "TextFeatureExtractor",
    "get_feature_extractor",
    "ChampionChallengerRouter",
    "get_champion_challenger_router",
    "PIIDetector",
    "PIIEntity",
    "get_pii_detector",
    "UnicodeSanitizer",
    "get_unicode_sanitizer",
    "DriftRecalibrationEngine",
    "RecalibrationResult",
    "get_recalibration_engine",
]
