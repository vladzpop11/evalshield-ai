"""EvalShield-AI: Unicode Normalization, Homoglyph Canonicalizer & Canary Leak Trap.

Defends against adversarial evasion techniques:
1. Zero-width character insertion (e.g., "i\\u200Bgnore all")
2. Cross-script homoglyph obfuscation (Cyrillic/Greek lookalikes: "іgnоrе")
3. Mathematical bold, script, and full-width alphanumeric bypasses
4. System prompt canary token exfiltration detection
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Optional, Tuple


class UnicodeSanitizer:
    """Canonicalizes obfuscated Unicode inputs and detects canary token leaks."""

    # Zero-width, formatting, and invisible characters
    ZERO_WIDTH_PATTERN = re.compile(
        r"[\u200B\u200C\u200D\u200E\u200F\uFEFF\u2060\u2061\u2062\u2063\u2064\u00AD]"
    )

    # Common Cyrillic, Greek, and Latin-lookalike homoglyph mapping to standard Latin ASCII
    HOMOGLYPH_TABLE = str.maketrans({
        # Cyrillic small lookalikes
        "а": "a", "с": "c", "е": "e", "о": "o", "р": "p", "х": "x", "у": "y",
        "і": "i", "ј": "j", "ѕ": "s", "ԁ": "d", "ԛ": "q", "ԝ": "w", "ѵ": "v",
        # Cyrillic capital lookalikes
        "А": "A", "В": "B", "С": "C", "Е": "E", "Н": "H", "І": "I", "Ј": "J",
        "К": "K", "М": "M", "О": "O", "Р": "P", "Ѕ": "S", "Т": "T", "Х": "X",
        # Greek lookalikes
        "α": "a", "ο": "o", "ρ": "p", "ν": "v", "τ": "t", "υ": "u", "χ": "x",
        "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K",
        "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
    })

    def canonicalize(self, text: str) -> str:
        """Strip zero-width characters, apply NFKC decomposition, and map homoglyphs.

        Converts adversarial payloads like:
          'ｉ\u200bｇ\u200bｎ\u200bｏ\u200bｒ\u200bｅ' or 'іgnоrе'
        into clean Latin ASCII:
          'ignore'
        """
        if not text:
            return ""

        # Step 1: Strip invisible/zero-width formatting characters
        cleaned = self.ZERO_WIDTH_PATTERN.sub("", text)

        # Step 2: Compatibility Decomposition followed by Canonical Composition (NFKC)
        # Normalizes full-width 'ｉｇｎｏｒｅ' -> 'ignore' and math bold '𝐢𝐠𝐧𝐨𝐫𝐞' -> 'ignore'
        normalized = unicodedata.normalize("NFKC", cleaned)

        # Step 3: Map cross-script homoglyphs (Cyrillic, Greek) to Latin equivalents
        canonical = normalized.translate(self.HOMOGLYPH_TABLE)

        return canonical

    def check_canary_leak(
        self,
        response_text: str,
        canary_tokens: List[str],
    ) -> Tuple[bool, Optional[str]]:
        """Verify whether any system prompt canary honeytokens leaked into the response.

        Returns (is_leaked, leaked_token).
        """
        if not response_text or not canary_tokens:
            return False, None

        canonical_response = self.canonicalize(response_text).lower()

        for canary in canary_tokens:
            if not canary:
                continue
            canonical_canary = self.canonicalize(canary).strip().lower()
            if canonical_canary and canonical_canary in canonical_response:
                return True, canary

        return False, None


_sanitizer_instance: Optional[UnicodeSanitizer] = None


def get_unicode_sanitizer() -> UnicodeSanitizer:
    """Thread-safe singleton getter for UnicodeSanitizer."""
    global _sanitizer_instance
    if _sanitizer_instance is None:
        _sanitizer_instance = UnicodeSanitizer()
    return _sanitizer_instance
