import pytest
import pytest_check as check

import numpy as np
from ipfx.stimulus import StimulusOntology
from ipfx.qc_feature_evaluator import DEFAULT_QC_CRITERIA_FILE

from qc_operator import QCOperator

# @pytest.fixture
# def sweep_data_tuple():
#     return (
#         {
#             'sweep_number': 0,
#             'response': np.empty([0,0]), 'stimulus': np.empty([0,0])
#         }
#     )
#
# @pytest.fixture
# def sweep_types():
#     return {'blowout': set()}
#
#
# @pytest.fixture
# def qc_operator():
#     return QCOperator(
#         sweep_data_tuple=sweep_data_tuple,
#         ontology=None,
#         qc_criteria={},
#         recording_date=""
#     )
#
# @pytest.mark.parametrize()
# def test_extract_blowout_mv():