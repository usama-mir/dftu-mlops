from tests import _PATH_DATA
import torch

def test_data():
    dataset_len = 25001
    train_data_shape = torch.load(f"{_PATH_DATA}/train_data.pt").shape
    train_labels_shape = torch.load(f"{_PATH_DATA}/train_labels.pt").shape
    test_data_shape = torch.load(f"{_PATH_DATA}/test_data.pt").shape
    test_labels_shape = torch.load(f"{_PATH_DATA}/test_labels.pt").shape

    assert train_data_shape[0] == dataset_len, f"Wrong dataset - doesn't match the required length. Length is {train_data_shape[0]}"
    assert torch.equal(torch.tensor(train_data_shape[1:]), torch.tensor([1, 28, 28])), f"Shape of images isn't correct. Shape is {train_data_shape[1:]}"