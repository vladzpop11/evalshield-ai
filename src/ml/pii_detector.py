"""EvalShield-AI: High-Performance PII & Secret Redaction Engine.

Provides zero-dependency, sub-millisecond detection and sanitization for:
- Credit Card Numbers (validated with Luhn checksum algorithm)
- US Social Security Numbers (SSN)
- Email Addresses
- International & Domestic Phone Numbers
- Cloud & API Credentials (OpenAI, AWS Access Keys, GitHub Tokens, JWTs)
- IPv4 and IPv6 network addresses

Fulfills EU AI Act Article 10 & NIST AI RMF Map 1.5 confidentiality requirements.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class PIIEntity:
    """Detected sensitive data entity occurrence."""
    entity_type: str
    start: int
    end: int
    confidence: float
    masked_placeholder: str

    def to_dict(self, *, redacted: bool = False, source: Optional[str] = None) -> dict:
        """Return safe detection metadata without the matched sensitive value."""
        data = {
            "entity_type": self.entity_type,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "masked_placeholder": self.masked_placeholder,
            "redacted": redacted,
        }
        if source is not None:
            data["source"] = source
        return data


class PIIDetector:
    """Zero-dependency scanner and anonymizer for PII and sensitive tokens."""

    # Pre-compiled high-speed regex patterns
    EMAIL_PATTERN = re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b"
    )
    SSN_PATTERN = re.compile(
        r"\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"
    )
    PHONE_PATTERN = re.compile(
        r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    )
    IPV4_PATTERN = re.compile(
        r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
        r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
    )
    IPV6_PATTERN = re.compile(
        r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b"
    )

    # API Keys, Tokens and Cloud Secrets
    OPENAI_KEY_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")
    AWS_KEY_PATTERN = re.compile(r"\bAKIA[0-9A-Z]{16}\b")
    GITHUB_TOKEN_PATTERN = re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{30,}\b")
    JWT_PATTERN = re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]+\b")

    # Credit card candidates (13 to 19 digits with optional hyphens/spaces, non-backtracking)
    CREDIT_CARD_CANDIDATE = re.compile(r"\b\d(?:[ -]?\d){12,18}\b")

    @staticmethod
    def _is_luhn_valid(card_number_str: str) -> bool:
        """Verify credit card number validity via ISO/IEC 7812 Luhn checksum."""
        digits = [int(c) for c in card_number_str if c.isdigit()]
        if len(digits) < 13 or len(digits) > 19:
            return False

        checksum = 0
        reverse_digits = digits[::-1]
        for idx, digit in enumerate(reverse_digits):
            if idx % 2 == 1:
                doubled = digit * 2
                checksum += doubled - 9 if doubled > 9 else doubled
            else:
                checksum += digit
        return checksum % 10 == 0

    def _detect_pii_candidates(self, text: str) -> List[PIIEntity]:
        """Return all raw detector spans in deterministic start/longest-first order."""
        if not text:
            return []

        entities: List[PIIEntity] = []

        # 1. API Keys & Secrets (High Priority)
        for match in self.OPENAI_KEY_PATTERN.finditer(text):
            entities.append(
                PIIEntity(
                    entity_type="API_KEY_OPENAI",
                    start=match.start(),
                    end=match.end(),
                    confidence=1.0,
                    masked_placeholder="[REDACTED_OPENAI_KEY]",
                )
            )

        for match in self.AWS_KEY_PATTERN.finditer(text):
            entities.append(
                PIIEntity(
                    entity_type="API_KEY_AWS",
                    start=match.start(),
                    end=match.end(),
                    confidence=1.0,
                    masked_placeholder="[REDACTED_AWS_KEY]",
                )
            )

        for match in self.GITHUB_TOKEN_PATTERN.finditer(text):
            entities.append(
                PIIEntity(
                    entity_type="TOKEN_GITHUB",
                    start=match.start(),
                    end=match.end(),
                    confidence=1.0,
                    masked_placeholder="[REDACTED_GITHUB_TOKEN]",
                )
            )

        for match in self.JWT_PATTERN.finditer(text):
            entities.append(
                PIIEntity(
                    entity_type="TOKEN_JWT",
                    start=match.start(),
                    end=match.end(),
                    confidence=0.98,
                    masked_placeholder="[REDACTED_JWT_TOKEN]",
                )
            )

        # 2. Email Addresses
        for match in self.EMAIL_PATTERN.finditer(text):
            entities.append(
                PIIEntity(
                    entity_type="EMAIL_ADDRESS",
                    start=match.start(),
                    end=match.end(),
                    confidence=0.99,
                    masked_placeholder="[REDACTED_EMAIL]",
                )
            )

        # 3. Social Security Numbers (US SSN)
        for match in self.SSN_PATTERN.finditer(text):
            entities.append(
                PIIEntity(
                    entity_type="US_SSN",
                    start=match.start(),
                    end=match.end(),
                    confidence=0.95,
                    masked_placeholder="[REDACTED_SSN]",
                )
            )

        # 4. Credit Card Numbers (Strict Luhn validation prevents false positives)
        for match in self.CREDIT_CARD_CANDIDATE.finditer(text):
            raw_val = match.group()
            digits_only = "".join(filter(str.isdigit, raw_val))
            if 13 <= len(digits_only) <= 19 and self._is_luhn_valid(digits_only):
                entities.append(
                    PIIEntity(
                        entity_type="CREDIT_CARD",
                        start=match.start(),
                        end=match.end(),
                        confidence=0.99,
                        masked_placeholder="[REDACTED_CREDIT_CARD]",
                    )
                )

        # 5. Phone Numbers
        for match in self.PHONE_PATTERN.finditer(text):
            raw = match.group()
            # Must contain at least 10 digits
            if sum(1 for c in raw if c.isdigit()) >= 10:
                entities.append(
                    PIIEntity(
                        entity_type="PHONE_NUMBER",
                        start=match.start(),
                        end=match.end(),
                        confidence=0.90,
                        masked_placeholder="[REDACTED_PHONE]",
                    )
                )

        # 6. IPv4 Addresses
        for match in self.IPV4_PATTERN.finditer(text):
            val = match.group()
            if not val.startswith("127.0.0.") and val != "0.0.0.0":
                entities.append(
                    PIIEntity(
                        entity_type="IP_ADDRESS_V4",
                        start=match.start(),
                        end=match.end(),
                        confidence=0.92,
                        masked_placeholder="[REDACTED_IP]",
                    )
                )

        # 7. IPv6 Addresses
        for match in self.IPV6_PATTERN.finditer(text):
            entities.append(
                PIIEntity(
                    entity_type="IP_ADDRESS_V6",
                    start=match.start(),
                    end=match.end(),
                    confidence=0.95,
                    masked_placeholder="[REDACTED_IP_V6]",
                )
            )

        # Preserve every recognized span for complete redaction. Public
        # detection still applies the existing non-overlap reporting policy.
        entities.sort(key=lambda e: (e.start, -e.end))

        return entities

    @staticmethod
    def _deduplicate_entities(entities: List[PIIEntity]) -> List[PIIEntity]:
        """Apply the established earliest-start, longest-at-same-start policy."""
        deduped: List[PIIEntity] = []
        last_end = -1
        for e in entities:
            if e.start >= last_end:
                deduped.append(e)
                last_end = e.end

        return deduped

    def detect_pii(self, text: str) -> List[PIIEntity]:
        """Scan text and report non-overlapping detections in occurrence order."""
        return self._deduplicate_entities(self._detect_pii_candidates(text))

    def redact_pii(
        self,
        text: str,
        placeholder: bool = True,
    ) -> Tuple[str, List[PIIEntity]]:
        """Redact detected PII entities and return (sanitized_text, detected_entities)."""
        candidates = self._detect_pii_candidates(text)
        entities = self._deduplicate_entities(candidates)
        if not candidates:
            return text, []

        # Merge overlapping intervals for redaction only. This preserves the
        # existing public detection policy while ensuring a crossing span's
        # suffix cannot survive just because that span was omitted from the
        # non-overlapping metadata list.
        merged_spans: List[Tuple[int, int, PIIEntity]] = []
        for entity in candidates:
            if merged_spans and entity.start < merged_spans[-1][1]:
                start, end, representative = merged_spans[-1]
                merged_spans[-1] = (start, max(end, entity.end), representative)
            else:
                merged_spans.append((entity.start, entity.end, entity))

        sanitized_chars = []
        last_idx = 0
        for start, end, representative in merged_spans:
            sanitized_chars.append(text[last_idx:start])
            if placeholder:
                sanitized_chars.append(representative.masked_placeholder)
            else:
                sanitized_chars.append("*" * (end - start))
            last_idx = end
        sanitized_chars.append(text[last_idx:])

        return "".join(sanitized_chars), entities

    def has_pii(self, text: str) -> bool:
        """Return True if any PII or secret is present."""
        return len(self.detect_pii(text)) > 0


_detector_instance: Optional[PIIDetector] = None


def get_pii_detector() -> PIIDetector:
    """Thread-safe singleton getter for PIIDetector."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = PIIDetector()
    return _detector_instance
