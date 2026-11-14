"""EvalShield-AI: PII & Sensitive Secret Detection Tests.

Validates:
- ISO/IEC 7812 Luhn checksum credit card validation
- US Social Security Number detection
- API key, token, and secret discovery
- Masking and sanitization functionality
"""

from __future__ import annotations

import pytest

from src.ml.pii_detector import PIIDetector, get_pii_detector


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
        assert entities[0].matched_value == "123-45-6789"

    def test_email_address_detection(self, detector: PIIDetector):
        """Corporate and personal email addresses are captured."""
        text = "Contact security team at sec-ops@enterprise-shield.corp for access."
        entities = detector.detect_pii(text)
        assert len(entities) == 1
        assert entities[0].entity_type == "EMAIL_ADDRESS"
        assert entities[0].matched_value == "sec-ops@enterprise-shield.corp"

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

    def test_clean_text_returns_unmodified(self, detector: PIIDetector):
        """Clean enterprise text without secrets remains completely untouched."""
        clean = "What are the core advantages of INT8 dynamic quantization in ONNX Runtime?"
        sanitized, entities = detector.redact_pii(clean)
        assert sanitized == clean
        assert len(entities) == 0
        assert detector.has_pii(clean) is False
