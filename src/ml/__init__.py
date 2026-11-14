"""EvalShield-AI Machine Learning Module."""

from .champion_challenger import ChampionChallengerRouter, get_champion_challenger_router
from .feature_extractor import TextFeatureExtractor, get_feature_extractor
from .inference import EvalShieldEngine, get_inference_engine
from .model_generator import generate_shield_model

__all__ = [
    "EvalShieldEngine",
    "get_inference_engine",
    "generate_shield_model",
    "TextFeatureExtractor",
    "get_feature_extractor",
    "ChampionChallengerRouter",
    "get_champion_challenger_router",
]
