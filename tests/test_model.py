import torch
from tests import _PATH_DATA
from src.models.model import Model
import os
import pytest


#@pytest.mark.skipif(not os.path(_PATH_DATA), reason="Data file not exist")
def test_model_output():
    expected_output = torch.tensor([1 for x in range(10)])
    expected_output = expected_output[None]
    test_model = Model()
    test_data = torch.load(f"{_PATH_DATA}/train_data.pt")[0]
    assert test_model(
        test_data.to(torch.float32)).shape == expected_output.shape, f"Wrong shape of output:{expected_output.shape}"
