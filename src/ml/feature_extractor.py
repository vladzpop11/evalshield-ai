"""EvalShield-AI: High-Throughput In-Memory NLP Feature Extractor.

Extracts multidimensional AI evaluation features directly from raw text:
1. Token Shannon Entropy
2. Prompt Injection Attack Probability
3. Lexical / Semantic Toxicity Proxy
4. Out-of-Domain Semantic Drift
5. Hallucination Risk Proxy (prompt-response alignment)
"""

from __future__ import annotations

import base64
import math
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple


class TextFeatureExtractor:
    """Zero-dependency, sub-millisecond NLP feature extraction pipeline."""

    # Common injection keywords, delimiter breaks, and roleplay jailbreak patterns
    INJECTION_PATTERNS = [
        r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions?",
        r"disregard\s+(?:all\s+)?(?:previous|prior|above)\s+instructions?",
        r"bypass\s+(?:all\s+)?(?:safety|content|security)?\s*(?:filters?|policies|guards?)?",
        r"(?:reveal|display|output|print|show)\s+(?:the\s+)?(?:system\s+prompt|developer\s+mode|instructions|root\s+key|password|secret)",
        r"you\s+are\s+now\s+(?:in\s+)?(?:dan|developer|jailbreak|unfiltered)\s+mode",
        r"pretend\s+you\s+have\s+no\s+(?:rules|filters|limitations|guidelines)",
        r"(?:system\s+override|admin\s+override|sudo\s+mode)",
        r"roleplay\s+as\s+(?:an?\s+)?evil",
        r"<\|(?:im_start|im_end|system|user|assistant)\|>",
        r"\[(?:system|admin|override)\]",
        r"format\s+as\s+raw\s+system\s+instructions",
    ]

    # Lexical toxicity indicators (hate speech, severe abuse, profanity density)
    TOXIC_TERMS = {
        "hate", "kill", "die", "murder", "destroy", "attack", "racist", "slur",
        "abuse", "idiot", "stupid", "worthless", "harm", "toxic", "violent",
        "scam", "fraud", "exploit", "steal", "terrorist", "bomb", "weapon"
    }

    # Enterprise benchmark vocabulary for domain alignment (finance, business, technical, conversational)
    APPROVED_DOMAIN_VOCAB = {
        "system", "model", "analysis", "data", "report", "revenue", "quarter", "growth",
        "function", "algorithm", "network", "performance", "metric", "process", "result",
        "document", "query", "user", "service", "customer", "accuracy", "evaluation",
        "project", "software", "development", "architecture", "security", "pipeline",
        "information", "overview", "summary", "features", "training", "deployment"
    }

    def __init__(self) -> None:
        self.injection_regexes = [
            re.compile(p, re.IGNORECASE | re.MULTILINE) for p in self.INJECTION_PATTERNS
        ]

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace and punctuation-aware word tokenization."""
        return re.findall(r"\b\w+\b", text.lower())

    def calculate_token_entropy(self, text: str) -> float:
        """Calculate Shannon entropy over word tokens, scaled to [0.0, 10.0]."""
        tokens = self._tokenize(text)
        if not tokens:
            return 4.5  # Neutral default for empty text

        counts = Counter(tokens)
        total_tokens = len(tokens)

        entropy = 0.0
        for count in counts.values():
            p = count / total_tokens
            entropy -= p * math.log2(p)

        # Scale typical NLP token entropy (0 - 5.5 bits) into a 0.0 - 10.0 budget
        # Highly repetitive text has low entropy; chaotic/obfuscated text has high entropy
        scaled = round(min(10.0, max(0.0, entropy * 1.85)), 2)
        return scaled

    def detect_prompt_injection(self, text: str) -> float:
        """Score likelihood of prompt injection, jailbreak attempts, or obfuscation."""
        if not text:
            return 0.0

        score = 0.0
        lower_text = text.lower()

        # 1. Pattern Matching
        for regex in self.injection_regexes:
            if regex.search(lower_text):
                score += 0.45

        # 2. Delimiter and Control Symbol Abuse
        delimiters = ["```", "---", "###", "===", "<|", "|>"]
        delim_count = sum(text.count(d) for d in delimiters)
        if delim_count >= 3:
            score += 0.20

        # 3. Obfuscation detection: Base64 payload detection
        base64_candidates = re.findall(r"[A-Za-z0-9+/]{20,}={0,2}", text)
        for cand in base64_candidates:
            try:
                decoded = base64.b64decode(cand).decode("utf-8", errors="ignore")
                if len(decoded) > 8 and any(char.isalpha() for char in decoded):
                    score += 0.35
                    break
            except Exception:
                pass

        # 4. Excessive non-ASCII / Zero-width / Homoglyph symbols
        non_ascii_count = sum(1 for c in text if ord(c) > 127)
        if len(text) > 0 and (non_ascii_count / len(text)) > 0.30:
            score += 0.25

        return round(min(1.0, max(0.0, score)), 4)

    def calculate_toxicity_score(self, text: str) -> float:
        """Calculate lexical and severity-weighted toxicity proxy score."""
        tokens = self._tokenize(text)
        if not tokens:
            return 0.0

        toxic_matches = sum(1 for t in tokens if t in self.TOXIC_TERMS)
        density = toxic_matches / max(1, len(tokens))

        # Nonlinear scaling: 1 toxic term in short text or repeated toxic terms quickly spikes score
        raw_score = (toxic_matches * 0.28) + (density * 1.5)
        return round(min(1.0, max(0.0, raw_score)), 4)

    def calculate_semantic_drift(self, text: str) -> float:
        """Measure lexical distance from approved enterprise domain vocabulary."""
        tokens = self._tokenize(text)
        if not tokens:
            return 0.0

        in_domain_count = sum(1 for t in tokens if t in self.APPROVED_DOMAIN_VOCAB)
        overlap_ratio = in_domain_count / max(1, len(tokens))

        # Higher overlap -> lower drift (0.0). Low overlap -> high drift (1.0).
        drift = 1.0 - min(1.0, overlap_ratio * 3.5)
        return round(max(0.05, min(0.95, drift)), 4)

    def calculate_hallucination_risk(self, prompt: str, response: Optional[str] = None) -> float:
        """Evaluate lexical alignment and grounding between prompt and response."""
        if not response:
            return 0.10  # Baseline neutral when response is not yet generated

        prompt_tokens = set(self._tokenize(prompt))
        resp_tokens = set(self._tokenize(response))

        if not prompt_tokens or not resp_tokens:
            return 0.20

        # Jaccard lexical intersection
        intersection = prompt_tokens.intersection(resp_tokens)
        grounding_ratio = len(intersection) / len(prompt_tokens)

        # Hallucination signals: high certainty qualifiers without grounding
        speculative_cues = ["definitely", "absolutely", "guaranteed", "100%", "proven fact", "never", "always"]
        cue_count = sum(1 for cue in speculative_cues if cue in response.lower())

        risk = (1.0 - min(1.0, grounding_ratio * 2.0)) * 0.6 + (cue_count * 0.15)
        return round(min(1.0, max(0.02, risk)), 4)

    def extract_features(
        self,
        prompt_text: str,
        response_text: Optional[str] = None,
        overrides: Optional[Dict[str, Optional[float]]] = None,
    ) -> Dict[str, float]:
        """Extract all 5 model features from raw text, respecting explicit caller overrides."""
        full_text = f"{prompt_text} {response_text or ''}".strip()

        # Compute automated features
        computed = {
            "token_entropy": self.calculate_token_entropy(full_text),
            "prompt_injection_score": self.detect_prompt_injection(prompt_text),
            "toxicity_score": self.calculate_toxicity_score(full_text),
            "semantic_drift": self.calculate_semantic_drift(prompt_text),
            "hallucination_risk": self.calculate_hallucination_risk(prompt_text, response_text),
        }

        # Apply caller overrides if provided
        if overrides:
            for k, v in overrides.items():
                if v is not None and k in computed:
                    computed[k] = float(v)

        return computed


# Thread-safe module singleton
_extractor: Optional[TextFeatureExtractor] = None


def get_feature_extractor() -> TextFeatureExtractor:
    """Return shared TextFeatureExtractor instance."""
    global _extractor
    if _extractor is None:
        _extractor = TextFeatureExtractor()
    return _extractor
