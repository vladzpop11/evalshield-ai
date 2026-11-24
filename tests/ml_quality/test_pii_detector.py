"""EvalShield-AI: PII & Sensitive Secret Detection Tests.

Validates:
- ISO/IEC 7812 Luhn checksum credit card validation
- US Social Security Number detection
- API key, token, and secret discovery
- Masking and sanitization functionality
"""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from src.ml.pii_detector import PIIEntity, PIIDetector, get_pii_detector


@pytest.fixture
def detector() -> PIIDetector:
    return get_pii_detector()


@pytest.mark.ml_quality
class TestPIIDetector:
    """Comprehensive test suite for PII and Secret Redaction Engine."""

    def test_singleton_retrieval(self, detector: PIIDetector):
        """get_pii_detector returns an active singleton instance."""
        assert detector is not None
        assert isinstance(detector, PIIDetector)

    def test_luhn_credit_card_detection(self, detector: PIIDetector):
        """Valid Luhn credit card numbers are detected; invalid sequences ignored."""
        # Standard test card numbers that pass Luhn algorithm
        valid_visa = "Please bill to 4532-0150-1234-5671 for this purchase."
        invalid_card = "My order number is 1234-5678-9012-3456 thank you."

        valid_entities = detector.detect_pii(valid_visa)
        assert len(valid_entities) == 1
        assert valid_entities[0].entity_type == "CREDIT_CARD"

        invalid_entities = detector.detect_pii(invalid_card)
        assert len(invalid_entities) == 0

    def test_us_ssn_detection(self, detector: PIIDetector):
        """Standard US Social Security Numbers are accurately identified."""
        text = "Employee record: John Doe, SSN: 123-45-6789, department engineering."
        entities = detector.detect_pii(text)
        assert len(entities) == 1
        assert entities[0].entity_type == "US_SSN"
        assert text[entities[0].start:entities[0].end] == "123-45-6789"

    def test_email_address_detection(self, detector: PIIDetector):
        """Corporate and personal email addresses are captured."""
        text = "Contact security team at sec-ops@enterprise-shield.corp for access."
        entities = detector.detect_pii(text)
        assert len(entities) == 1
        assert entities[0].entity_type == "EMAIL_ADDRESS"
        assert text[entities[0].start:entities[0].end] == "sec-ops@enterprise-shield.corp"

    def test_entity_serialization_and_repr_never_expose_matched_value(self, detector: PIIDetector):
        secret = "private.user@example.com"
        entity = detector.detect_pii(secret)[0]

        serialized = json.dumps(entity.to_dict())
        dataclass_serialized = json.dumps(asdict(entity))
        assert secret not in serialized
        assert secret not in dataclass_serialized
        assert "matched_value" not in entity.to_dict()
        assert secret not in repr(entity)
        assert not hasattr(entity, "matched_value")
        assert entity.to_dict()["entity_type"] == "EMAIL_ADDRESS"
        assert entity.to_dict()["redacted"] is False

    def test_api_keys_and_secrets_detection(self, detector: PIIDetector):
        """OpenAI, AWS, GitHub, and JWT credentials trigger immediate discovery."""
        sample_openai = "Here is my key: sk-abcdef1234567890abcdef1234567890"
        sample_aws = "Deploy with access key AKIAIOSFODNN7EXAMPLE today."
        sample_gh = "GitHub sync token ghp_1234567890abcdefghijklmnopqrstuvwxyz"

        res_openai = detector.detect_pii(sample_openai)
        assert len(res_openai) == 1
        assert res_openai[0].entity_type == "API_KEY_OPENAI"

        res_aws = detector.detect_pii(sample_aws)
        assert len(res_aws) == 1
        assert res_aws[0].entity_type == "API_KEY_AWS"

        res_gh = detector.detect_pii(sample_gh)
        assert len(res_gh) == 1
        assert res_gh[0].entity_type == "TOKEN_GITHUB"

    def test_phone_and_ipv4_detection(self, detector: PIIDetector):
        """Phone numbers and external IP addresses are recognized."""
        text = "Call +1-800-555-0199 or connect via 198.51.100.42 gateway."
        entities = detector.detect_pii(text)
        types = [e.entity_type for e in entities]
        assert "PHONE_NUMBER" in types
        assert "IP_ADDRESS_V4" in types

    def test_redact_pii_replaces_with_placeholders(self, detector: PIIDetector):
        """Redact method safely scrubs sensitive values into readable placeholders."""
        raw = "User alice@company.com with phone 555-123-4567 reported issue."
        sanitized, entities = detector.redact_pii(raw)

        assert len(entities) == 2
        assert "alice@company.com" not in sanitized
        assert "555-123-4567" not in sanitized
        assert "[REDACTED_EMAIL]" in sanitized
        assert "[REDACTED_PHONE]" in sanitized
        assert entities[0].to_dict(redacted=True)["redacted"] is True

    def test_overlapping_credit_card_and_phone_spans_keep_longest_detection(self, detector: PIIDetector):
        # The 16-digit card also contains a 10-digit substring matching the
        # phone detector. Same-start overlaps follow the existing longest-span policy.
        card = "4532015012345671"
        entities = detector.detect_pii(card)
        assert [(entity.entity_type, entity.start, entity.end) for entity in entities] == [
            ("CREDIT_CARD", 0, len(card))
        ]
        sanitized, redacted_entities = detector.redact_pii(card)
        assert sanitized == "[REDACTED_CREDIT_CARD]"
        assert len(redacted_entities) == 1

    def test_crossing_spans_are_fully_redacted_without_changing_detection_policy(self):
        class CrossingSpanDetector(PIIDetector):
            def _detect_pii_candidates(self, text):
                # Controlled spans [0, 6) and [3, 9): this exercises interval
                # handling without claiming a built-in regex creates this pair.
                return [
                    PIIEntity("TEST_A", 0, 6, 1.0, "[REDACTED_A]"),
                    PIIEntity("TEST_B", 3, 9, 1.0, "[REDACTED_B]"),
                ]

        crossing_detector = CrossingSpanDetector()
        source = "ABCDEFGHIJ"
        # Public metadata retains the established earliest-start policy.
        assert [entity.entity_type for entity in crossing_detector.detect_pii(source)] == ["TEST_A"]
        sanitized, entities = crossing_detector.redact_pii(source)
        assert sanitized == "[REDACTED_A]J"
        assert len(entities) == 1

        masked, _ = crossing_detector.redact_pii(source, placeholder=False)
        assert masked == "*********J"

    def test_adjacent_spans_are_both_detected_and_redacted(self, detector: PIIDetector):
        first_phone = "5551234567"
        second_phone = "(555)123-4567"
        raw = first_phone + second_phone
        entities = detector.detect_pii(raw)
        assert [(entity.entity_type, entity.start, entity.end) for entity in entities] == [
            ("PHONE_NUMBER", 0, len(first_phone)),
            ("PHONE_NUMBER", len(first_phone), len(raw)),
        ]
        sanitized, _ = detector.redact_pii(raw)
        assert sanitized == "[REDACTED_PHONE][REDACTED_PHONE]"
        assert first_phone not in sanitized
        assert second_phone not in sanitized

    def test_redact_without_placeholder_masks_each_span_length(self, detector: PIIDetector):
        raw = "alice@example.com and 4532015012345671"
        entities = detector.detect_pii(raw)
        sanitized, _ = detector.redact_pii(raw, placeholder=False)

        expected = raw
        for entity in reversed(entities):
            expected = expected[:entity.start] + ("*" * (entity.end - entity.start)) + expected[entity.end:]
        assert sanitized == expected
        assert "alice@example.com" not in sanitized
        assert "4532015012345671" not in sanitized

    def test_clean_text_returns_unmodified(self, detector: PIIDetector):
        """Clean enterprise text without secrets remains completely untouched."""
        clean = "What are the core advantages of INT8 dynamic quantization in ONNX Runtime?"
        sanitized, entities = detector.redact_pii(clean)
        assert sanitized == clean
        assert len(entities) == 0
        assert detector.has_pii(clean) is False
