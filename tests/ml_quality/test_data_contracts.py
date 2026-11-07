"""EvalShield-AI: ML Data Contract Tests.

Uses Pandera to validate schema integrity, range boundaries, nullability,
and data types for training benchmarks and live production feeds.
"""

import numpy as np
import pandas as pd
try:
    import pandera.pandas as pa
    from pandera.pandas import Check, Column, DataFrameSchema
except (ImportError, AttributeError):
    import pandera as pa
    from pandera import Check, Column, DataFrameSchema
import pytest

from src.ml.model_generator import generate_synthetic_benchmark


# Define the strict Pandera Data Contract
EvalShieldInputSchema = DataFrameSchema(
    columns={
        "toxicity_score": Column(
            float,
            checks=[
                Check.greater_than_or_equal_to(0.0),
                Check.less_than_or_equal_to(1.0),
            ],
            nullable=False,
            coerce=True,
        ),
        "hallucination_risk": Column(
            float,
            checks=[
                Check.greater_than_or_equal_to(0.0),
                Check.less_than_or_equal_to(1.0),
            ],
            nullable=False,
            coerce=True,
        ),
        "prompt_injection_score": Column(
            float,
            checks=[
                Check.greater_than_or_equal_to(0.0),
                Check.less_than_or_equal_to(1.0),
            ],
            nullable=False,
            coerce=True,
        ),
        "semantic_drift": Column(
            float,
            checks=[
                Check.greater_than_or_equal_to(0.0),
                Check.less_than_or_equal_to(1.0),
            ],
            nullable=False,
            coerce=True,
        ),
        "token_entropy": Column(
            float,
            checks=[
                Check.greater_than_or_equal_to(0.0),
                Check.less_than_or_equal_to(10.0),
            ],
            nullable=False,
            coerce=True,
        ),
    },
    strict=True,
    ordered=False,
)

EvalShieldTrainingSchema = DataFrameSchema(
    columns={
        **EvalShieldInputSchema.columns,
        "shield_decision": Column(
            int,
            checks=[Check.isin([0, 1, 2])],
            nullable=False,
            coerce=True,
        ),
    },
    strict=True,
    ordered=False,
)


@pytest.mark.ml_quality
class TestDataContracts:
    """Validate data schemas and contracts against Pandera specifications."""

    def test_benchmark_data_adheres_to_contract(self):
        """Verify that synthetic training generator strictly satisfies Pandera contract."""
        df, labels = generate_synthetic_benchmark(n_samples=500, random_state=123)
        combined = df.assign(shield_decision=labels)

        # Should validate without error
        validated_df = EvalShieldTrainingSchema.validate(combined)
        assert isinstance(validated_df, pd.DataFrame)
        assert len(validated_df) == 500

    def test_missing_required_column_fails(self):
        """Verify that dropping a feature column causes SchemaError."""
        df, _ = generate_synthetic_benchmark(n_samples=50, random_state=42)
        invalid_df = df.drop(columns=["toxicity_score"])

        with pytest.raises((pa.errors.SchemaError, pa.errors.SchemaErrors)):
            EvalShieldInputSchema.validate(invalid_df)

    def test_unexpected_extra_column_fails_strictness(self):
        """Verify that strict mode rejects unapproved extraneous columns."""
        df, _ = generate_synthetic_benchmark(n_samples=50, random_state=42)
        invalid_df = df.assign(unauthorized_field="malicious_payload")

        with pytest.raises((pa.errors.SchemaError, pa.errors.SchemaErrors)):
            EvalShieldInputSchema.validate(invalid_df)

    @pytest.mark.parametrize(
        "column,invalid_value",
        [
            ("toxicity_score", 1.05),
            ("toxicity_score", -0.01),
            ("hallucination_risk", 2.0),
            ("prompt_injection_score", -0.5),
            ("semantic_drift", 1.2),
            ("token_entropy", 10.5),
            ("token_entropy", -0.1),
        ],
    )
    def test_out_of_bound_values_fail_contract(self, column: str, invalid_value: float):
        """Verify that values breaching physical/mathematical domain boundaries fail."""
        df, _ = generate_synthetic_benchmark(n_samples=20, random_state=42)
        df.loc[0, column] = invalid_value

        with pytest.raises((pa.errors.SchemaError, pa.errors.SchemaErrors)):
            EvalShieldInputSchema.validate(df)

    def test_null_values_rejected(self):
        """Verify that null/NaN values are strictly rejected by the contract."""
        df, _ = generate_synthetic_benchmark(n_samples=20, random_state=42)
        df.loc[3, "hallucination_risk"] = np.nan

        with pytest.raises((pa.errors.SchemaError, pa.errors.SchemaErrors)):
            EvalShieldInputSchema.validate(df)
