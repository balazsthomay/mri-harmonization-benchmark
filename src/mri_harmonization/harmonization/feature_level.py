"""Feature-level harmonization using ComBat."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ComBatHarmonizer:
    """ComBat batch-effect correction for radiomics features.

    Applies Empirical Bayes harmonization to remove site effects while
    preserving biological covariates (age, sex).
    """

    def __init__(
        self,
        batch_col: str = "site",
        categorical_cols: list[str] | None = None,
        continuous_cols: list[str] | None = None,
    ) -> None:
        self._batch_col = batch_col
        self._categorical_cols = categorical_cols or []
        self._continuous_cols = continuous_cols or []

    def harmonize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply ComBat harmonization to a feature DataFrame.

        Args:
            df: DataFrame with subject_id, site, and feature columns.
                May also contain biological covariate columns.

        Returns:
            DataFrame with harmonized feature values, same shape as input.
        """
        from neuroCombat import neuroCombat

        # Identify feature columns (exclude metadata and covariates)
        meta_cols = {"subject_id", self._batch_col}
        meta_cols.update(self._categorical_cols)
        meta_cols.update(self._continuous_cols)
        feature_cols = [c for c in df.columns if c not in meta_cols]

        # Prepare data matrix (features x subjects, as neuroCombat expects)
        data = df[feature_cols].values.T

        # Prepare covariates
        covars = pd.DataFrame({self._batch_col: df[self._batch_col].values})
        for col in self._categorical_cols:
            covars[col] = df[col].values
        for col in self._continuous_cols:
            covars[col] = df[col].values

        # Run ComBat
        combat_result = neuroCombat(
            dat=data,
            covars=covars,
            batch_col=self._batch_col,
            categorical_cols=self._categorical_cols if self._categorical_cols else None,
            continuous_cols=self._continuous_cols if self._continuous_cols else None,
        )

        harmonized_data = combat_result["data"]

        # Reconstruct DataFrame
        result = df.copy()
        result[feature_cols] = harmonized_data.T

        logger.info(
            "ComBat harmonization complete: %d features, %d subjects",
            len(feature_cols),
            len(df),
        )

        return result
