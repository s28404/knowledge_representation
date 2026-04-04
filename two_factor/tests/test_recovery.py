"""Unit tests for two_factor.recovery module."""

import pytest

from two_factor.recovery import (
    generate_recovery_codes,
    hash_recovery_code,
    verify_recovery_code,
)


class TestGenerateRecoveryCodes:
    def test_default_count(self):
        codes = generate_recovery_codes()
        assert len(codes) == 10

    def test_custom_count(self):
        codes = generate_recovery_codes(count=5)
        assert len(codes) == 5

    def test_code_format(self):
        codes = generate_recovery_codes(count=20)
        for code in codes:
            parts = code.split("-")
            assert len(parts) == 4, f"Expected 4 groups, got {len(parts)}: {code}"
            for part in parts:
                assert len(part) == 5, f"Expected 5 chars per group: {part}"
                assert part.isalnum(), f"Non-alphanumeric characters in group: {part}"

    def test_codes_are_unique(self):
        codes = generate_recovery_codes(count=50)
        assert len(set(codes)) == len(codes), "Duplicate recovery codes generated"

    def test_codes_are_uppercase(self):
        codes = generate_recovery_codes(count=10)
        for code in codes:
            assert code == code.upper(), f"Code contains lower-case characters: {code}"


class TestHashRecoveryCode:
    def test_returns_string(self):
        h = hash_recovery_code("ABCDE-FGHJK-MNPQR-STUV2")
        assert isinstance(h, str)

    def test_correct_length(self):
        # 32 hex chars (salt) + 64 hex chars (SHA-256) = 96
        h = hash_recovery_code("ABCDE-FGHJK-MNPQR-STUV2")
        assert len(h) == 96

    def test_different_salts_per_call(self):
        code = "ABCDE-FGHJK-MNPQR-STUV2"
        h1 = hash_recovery_code(code)
        h2 = hash_recovery_code(code)
        assert h1 != h2, "Same code should produce different hashes due to random salt"

    def test_only_hex_characters(self):
        h = hash_recovery_code("ABCDE-FGHJK-MNPQR-STUV2")
        assert all(c in "0123456789abcdef" for c in h), "Hash contains non-hex characters"


class TestVerifyRecoveryCode:
    def _setup(self, count=5):
        codes = generate_recovery_codes(count=count)
        stored = [hash_recovery_code(c) for c in codes]
        return codes, stored

    def test_valid_code_returns_true(self):
        codes, stored = self._setup()
        success, _ = verify_recovery_code(stored, codes[0])
        assert success is True

    def test_invalid_code_returns_false(self):
        _, stored = self._setup()
        success, remaining = verify_recovery_code(stored, "AAAAA-BBBBB-CCCCC-DDDDD")
        assert success is False
        assert remaining == stored

    def test_used_code_is_removed(self):
        codes, stored = self._setup(count=5)
        success, remaining = verify_recovery_code(stored, codes[2])
        assert success is True
        assert len(remaining) == 4

    def test_code_cannot_be_reused(self):
        codes, stored = self._setup(count=3)
        _, updated = verify_recovery_code(stored, codes[0])
        success, _ = verify_recovery_code(updated, codes[0])
        assert success is False

    def test_case_insensitive_match(self):
        codes, stored = self._setup(count=3)
        lowered = codes[0].lower()
        success, _ = verify_recovery_code(stored, lowered)
        assert success is True

    def test_whitespace_trimmed(self):
        codes, stored = self._setup(count=3)
        padded = "  " + codes[1] + "  "
        success, _ = verify_recovery_code(stored, padded)
        assert success is True

    def test_remaining_codes_unchanged_after_invalid(self):
        codes, stored = self._setup(count=5)
        success, remaining = verify_recovery_code(stored, "XXXXX-XXXXX-XXXXX-XXXXX")
        assert success is False
        assert remaining == stored

    def test_all_codes_can_be_consumed_sequentially(self):
        codes, stored = self._setup(count=3)
        for code in codes:
            success, stored = verify_recovery_code(stored, code)
            assert success is True
        assert stored == []

    def test_empty_stored_list(self):
        success, remaining = verify_recovery_code([], "ABCDE-FGHJK-MNPQR-STUV2")
        assert success is False
        assert remaining == []
