from unittest import mock
from unittest.mock import MagicMock
import os
import torch
import pytest
from Code.Application.project_name_model import (ProjectNameModel, cosine_similarity)


cwd = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = 'test_cnc_ext.csv'
FIXTURES_PATH = os.path.join(cwd, 'Fixtures', 'CNC')
CNC_PATH = os.path.join(cwd, 'Fixtures', 'CNC', CSV_FILE)
MODEL_NAME = "microsoft/codebert-base"


@mock.patch("Code.Application.project_name_model.read_cnc_csv")
def test_init_project_name_model(mock_read_csv: mock) -> None:
    mock_read_csv.return_value = "test"
    model = ProjectNameModel(CNC_PATH)
    assert model.cnc_df == "test"
    assert model.knowledge_tokenized == []
    assert model.knowledge_cnc_name == []
    assert model.pretrained_model is None
    assert model.pretrained_tokenizer is None
    mock_read_csv.assert_called_once()


@mock.patch("Code.Application.project_name_model.AutoModel")
@mock.patch("Code.Application.project_name_model.AutoTokenizer")
@mock.patch("Code.Application.project_name_model.logging")
def test_load_pretrained_llm_model_none_tokenizer_none(mock_logging: mock,
                                                       mock_autotockenizer: mock,
                                                       mock_automodel: mock) -> None:
    mock_autotockenizer.from_pretrained.return_value = "tockenizer"
    mock_automodel.from_pretrained.return_value = "model"
    model = ProjectNameModel(CNC_PATH)
    model.load_pretrained_llm(MODEL_NAME)
    mock_autotockenizer.from_pretrained.assert_called_with(MODEL_NAME)
    mock_automodel.from_pretrained.assert_called_with(MODEL_NAME)
    mock_logging.info.assert_called_with(f'{MODEL_NAME} models loaded')
    assert model.pretrained_model == "model"
    assert model.pretrained_tokenizer == "tockenizer"


@mock.patch("Code.Application.project_name_model.AutoModel")
@mock.patch("Code.Application.project_name_model.AutoTokenizer")
@mock.patch("Code.Application.project_name_model.logging")
def test_load_pretrained_llm_tokenizer_none(mock_logging: mock,
                                            mock_autotockenizer: mock,
                                            mock_automodel: mock) -> None:
    mock_autotockenizer.from_pretrained.return_value = "tockenizer"
    mock_automodel.from_pretrained.return_value = "model"
    model = ProjectNameModel(CNC_PATH)
    model.pretrained_model = "model"
    model.pretrained_tokenizer = None
    model.load_pretrained_llm(MODEL_NAME)
    mock_automodel.from_pretrained.assert_not_called()
    mock_autotockenizer.from_pretrained.assert_called_with(MODEL_NAME)
    mock_logging.info.assert_called_with(f'{MODEL_NAME} models loaded')
    assert model.pretrained_model == "model"
    assert model.pretrained_tokenizer == "tockenizer"


@mock.patch("Code.Application.project_name_model.AutoModel")
@mock.patch("Code.Application.project_name_model.AutoTokenizer")
@mock.patch("Code.Application.project_name_model.logging")
def test_load_pretrained_llm_model_none(mock_logging: mock,
                                        mock_autotockenizer: mock,
                                        mock_automodel: mock) -> None:
    mock_autotockenizer.from_pretrained.return_value = "tockenizer"
    mock_automodel.from_pretrained.return_value = "model"
    model = ProjectNameModel(CNC_PATH)
    model.pretrained_model = None
    model.pretrained_tokenizer = "tockenizer"
    model.load_pretrained_llm(MODEL_NAME)
    mock_autotockenizer.from_pretrained.assert_not_called()
    mock_automodel.from_pretrained.assert_called_with(MODEL_NAME)
    mock_logging.info.assert_called_with(f'{MODEL_NAME} models loaded')
    assert model.pretrained_model == "model"
    assert model.pretrained_tokenizer == "tockenizer"


@pytest.fixture
def vectors():
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y = torch.tensor([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
    return x, y


@mock.patch("Code.Application.project_name_model.logging")
def test_cosine_similarity(mock_logging: mock, vectors):
    x, y = vectors
    similarity = cosine_similarity(x, y)
    assert similarity.shape == (2, 2)
    assert torch.allclose(similarity, torch.tensor([[0.9926, 1.8859], [0.5107, 0.9996]]), atol=1e-4)
    mock_logging.info.assert_called_with('cosine_similarity end')


@mock.patch("Code.Application.project_name_model.logging")
@mock.patch("Code.Application.project_name_model.cosine_similarity")
def test_compare_documents(mock_cosine_similarity: mock,
                           mock_logging: mock) -> None:
    doc1 = 'ABC'
    doc2 = 'CDE'
    model = ProjectNameModel(CNC_PATH)
    model.pretrained_tokenizer = MagicMock()
    model.pretrained_tokenizer().input_ids = 'test'
    model.pretrained_model = MagicMock()
    model.compare_documents(doc1, doc2)
    model.pretrained_tokenizer.assert_called_with(doc2, return_tensors="pt", max_length=512)
    model.pretrained_model.assert_called_with('test')
    mock_cosine_similarity.assert_called_once()
    mock_logging.info.assert_called_with('compare_documents finished')
