"""Integration tests for the ECME estimator."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Literal

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pysatl_mpest.core import MixtureModel
from pysatl_mpest.estimators import ECME
from pysatl_mpest.estimators.iterative import PriorThresholdPruner, StepBreakpointer
from pysatl_mpest.optimizers import ScipyNelderMead

from tests.helpers.estimator_data import (
    PRUNING_THRESHOLD,
    TRUE_WEIGHTS,
    ecm_for,
    exponential,
    fitted_arrays,
    parameters,
    stray_start,
)


class TestStrategies:
    def test_with_q_function_strategy_equals_ecm(self, X: np.ndarray, start: MixtureModel) -> None:
        """Compare the ECME fit with ECM as the second estimator."""
        ecme = ECME(
            breakpointers=[StepBreakpointer(10)], pruners=[], optimizer=ScipyNelderMead(), default_strategy="q-func"
        )

        assert ecme.fit(X, start) == ecm_for(10).fit(X, start)

    @pytest.mark.parametrize(
        ("default_strategy", "q_indices", "odl_indices"),
        [("odl", None, None), ("q-func", None, 1), ("q-func", 0, 1)],
        ids=["odl everywhere", "odl for the second", "q-function and odl"],
    )
    def test_with_observed_data_likelihood_improves_the_fit(
        self,
        X: np.ndarray,
        start: MixtureModel,
        default_strategy: Literal["q-func", "odl"],
        q_indices: int | None,
        odl_indices: int | None,
    ) -> None:
        ecme = ECME(
            breakpointers=[StepBreakpointer(25)],
            pruners=[],
            optimizer=ScipyNelderMead(),
            default_strategy=default_strategy,
        )

        fitted = ecme.fit(X, start, q_indices_raw=q_indices, odl_indices_raw=odl_indices)

        assert fitted.loglikelihood(X) > start.loglikelihood(X)
        mu, sigma, weights = fitted_arrays(fitted)
        assert_allclose(mu, [0.0, 6.0], atol=0.2)
        assert_allclose(sigma, [1.0, 1.5], atol=0.2)
        assert_allclose(weights, TRUE_WEIGHTS, atol=0.05)
        assert len(ecme.history) == 25  # noqa: PLR2004


class TestUnsupportedData:
    def test_with_an_unsupported_point_matches_the_fit_on_clean_data(self) -> None:
        truth = MixtureModel([exponential(1.5), exponential(6.0)], weights=[0.4, 0.6])
        clean = np.asarray(truth.generate(size=600, random_state=42), dtype=np.float64)
        start = MixtureModel([exponential(0.5), exponential(3.0)])

        def estimator() -> ECME:
            """Create a fresh estimator with independent stopping state."""
            return ECME(
                breakpointers=[StepBreakpointer(50)],
                pruners=[],
                optimizer=ScipyNelderMead(),
                default_strategy="odl",
            )

        clean_fit = estimator().fit(clean, start)
        outlier_fit = estimator().fit(np.append(clean, -0.5), start)

        for actual, expected in zip(parameters(outlier_fit), parameters(clean_fit), strict=True):
            assert actual == pytest.approx(expected, rel=1e-3)
        assert_allclose(outlier_fit.weights, clean_fit.weights, rtol=1e-3)
        assert not np.allclose(
            [component["lambda_"] for component in parameters(outlier_fit)],
            [component["lambda_"] for component in parameters(start)],
        )


class TestPruning:
    @pytest.mark.parametrize("default_strategy", ["q-func", "odl"])
    def test_removes_a_component_with_a_negligible_weight(
        self, X: np.ndarray, default_strategy: Literal["q-func", "odl"]
    ) -> None:
        ecme = ECME(
            breakpointers=[StepBreakpointer(12)],
            pruners=[PriorThresholdPruner(PRUNING_THRESHOLD)],
            optimizer=ScipyNelderMead(),
            default_strategy=default_strategy,
        )

        fitted = ecme.fit(X, stray_start((0,)))

        assert fitted.n_components == 2  # noqa: PLR2004
        assert_allclose(fitted_arrays(fitted)[0], [0.0, 6.0], atol=0.2)
        assert_allclose(fitted.weights, TRUE_WEIGHTS, atol=0.05)
