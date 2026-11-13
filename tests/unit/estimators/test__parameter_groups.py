"""
Unit tests for the parameter-groups validation and block-chain helpers of the ECM and ECME facades.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pytest
from pysatl_core.types import FamilyName
from pysatl_mpest.core import MixtureModel
from pysatl_mpest.estimators._parameter_groups import component_block_chain, validate_parameter_groups
from pysatl_mpest.estimators.iterative import MaximizationStrategy, OptimizationBlock

from tests.helpers.families import family


@pytest.fixture
def two_component_mixture() -> MixtureModel:
    """A mixture whose component 0 has free parameters ``mu`` and ``sigma`` and component 1 has ``lambda_``."""
    return MixtureModel(
        [
            family(FamilyName.NORMAL)(mu=0.0, sigma=1.0),
            family(FamilyName.EXPONENTIAL)(lambda_=1.0),
        ],
        weights=[0.4, 0.6],
    )


@pytest.fixture
def partially_fixed_mixture() -> MixtureModel:
    """A mixture whose component 0 has ``mu`` fixed by a view and only ``sigma`` free."""
    return MixtureModel(
        [
            family(FamilyName.NORMAL).view(mu=0.0)(sigma=1.0),
            family(FamilyName.EXPONENTIAL)(lambda_=1.0),
        ],
        weights=[0.4, 0.6],
    )


class TestValidateParameterGroups:
    def test_none_and_empty_mapping_are_accepted(self, two_component_mixture: MixtureModel):
        validate_parameter_groups(two_component_mixture, None)
        validate_parameter_groups(two_component_mixture, {})

    def test_valid_partitions_are_accepted(self, two_component_mixture: MixtureModel):
        validate_parameter_groups(two_component_mixture, {0: [{"mu"}, {"sigma"}], 1: [{"lambda_"}]})
        validate_parameter_groups(two_component_mixture, {0: [{"mu", "sigma"}]})
        validate_parameter_groups(two_component_mixture, {1: [{"lambda_"}]})

    def test_negative_key_is_rejected(self, two_component_mixture: MixtureModel):
        with pytest.raises(ValueError, match="negative"):
            validate_parameter_groups(two_component_mixture, {-1: [{"mu"}]})

    @pytest.mark.parametrize("key", [2, 10], ids=["at-bound", "above-bound"])
    def test_out_of_bounds_key_is_rejected(self, two_component_mixture: MixtureModel, key: int):
        with pytest.raises(ValueError, match="out of bounds"):
            validate_parameter_groups(two_component_mixture, {key: [{"mu"}]})

    def test_empty_group_list_is_rejected(self, two_component_mixture: MixtureModel):
        with pytest.raises(ValueError, match="must not be empty"):
            validate_parameter_groups(two_component_mixture, {0: []})

    def test_empty_group_is_rejected(self, two_component_mixture: MixtureModel):
        with pytest.raises(ValueError, match="empty group"):
            validate_parameter_groups(two_component_mixture, {0: [{"mu"}, set()]})

    def test_overlapping_groups_are_rejected(self, two_component_mixture: MixtureModel):
        with pytest.raises(ValueError, match="overlap"):
            validate_parameter_groups(two_component_mixture, {0: [{"mu", "sigma"}, {"sigma"}]})

    def test_uncovered_parameters_are_rejected(self, two_component_mixture: MixtureModel):
        with pytest.raises(ValueError, match="not covered"):
            validate_parameter_groups(two_component_mixture, {0: [{"mu"}]})

    def test_unknown_parameters_are_rejected(self, two_component_mixture: MixtureModel):
        with pytest.raises(ValueError, match="are not parameters"):
            validate_parameter_groups(two_component_mixture, {0: [{"mu", "sigma", "x"}]})

    def test_fixed_parameters_are_rejected(self, partially_fixed_mixture: MixtureModel):
        with pytest.raises(ValueError, match="are fixed and cannot be optimized"):
            validate_parameter_groups(partially_fixed_mixture, {0: [{"mu"}, {"sigma"}]})

    def test_a_group_covering_only_free_parameters_of_a_view_is_accepted(self, partially_fixed_mixture: MixtureModel):
        validate_parameter_groups(partially_fixed_mixture, {0: [{"sigma"}], 1: [{"lambda_"}]})


class TestComponentBlockChain:
    def test_no_groups_gives_a_single_joint_block(self):
        blocks = component_block_chain(1, None, MaximizationStrategy.QFUNCTION)

        assert blocks == [OptimizationBlock(1, None, MaximizationStrategy.QFUNCTION)]

    def test_groups_give_ordered_blocks_of_one_component(self):
        blocks = component_block_chain(0, [{"mu"}, {"sigma"}], MaximizationStrategy.QFUNCTION)

        assert blocks == [
            OptimizationBlock(0, {"mu"}, MaximizationStrategy.QFUNCTION),
            OptimizationBlock(0, {"sigma"}, MaximizationStrategy.QFUNCTION),
        ]

    def test_the_strategy_is_applied_to_every_block(self):
        blocks = component_block_chain(2, [{"a"}, {"b"}], MaximizationStrategy.OBSERVED_DATA_LIKELIHOOD)

        assert all(block.maximization_strategy == MaximizationStrategy.OBSERVED_DATA_LIKELIHOOD for block in blocks)

    def test_blocks_get_copies_of_the_groups(self):
        group = {"mu"}
        blocks = component_block_chain(0, [group], MaximizationStrategy.QFUNCTION)

        assert blocks[0].params_to_optimize == group
        assert blocks[0].params_to_optimize is not group
