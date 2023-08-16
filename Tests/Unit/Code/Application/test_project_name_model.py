from unittest import mock
import os
from Code.Application.project_name_model import ProjectNameModel


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
def test_load_pretrained_llm_model_none_tokenizer_none(mock_autotockenizer: mock,
                                                       mock_automodel: mock) -> None:
    mock_autotockenizer.from_pretrained.return_value = "tockenizer"
    mock_automodel.from_pretrained.return_value = "model"
    model = ProjectNameModel(CNC_PATH)
    model.load_pretrained_llm(MODEL_NAME)
    mock_autotockenizer.from_pretrained.assert_called_with(MODEL_NAME)
    mock_automodel.from_pretrained.assert_called_with(MODEL_NAME)
    assert model.pretrained_model == "model"
    assert model.pretrained_tokenizer == "tockenizer"


@mock.patch("Code.Application.project_name_model.AutoModel")
@mock.patch("Code.Application.project_name_model.AutoTokenizer")
def test_load_pretrained_llm_tokenizer_none(mock_autotockenizer: mock,
                                            mock_automodel: mock) -> None:
    mock_autotockenizer.from_pretrained.return_value = "tockenizer"
    mock_automodel.from_pretrained.return_value = "model"
    model = ProjectNameModel(CNC_PATH)
    model.pretrained_model = "model"
    model.pretrained_tokenizer = None
    model.load_pretrained_llm(MODEL_NAME)
    mock_automodel.from_pretrained.assert_not_called()
    mock_autotockenizer.from_pretrained.assert_called_with(MODEL_NAME)
    assert model.pretrained_model == "model"
    assert model.pretrained_tokenizer == "tockenizer"


@mock.patch("Code.Application.project_name_model.AutoModel")
@mock.patch("Code.Application.project_name_model.AutoTokenizer")
def test_load_pretrained_llm_model_none(mock_autotockenizer: mock,
                                            mock_automodel: mock) -> None:
    mock_autotockenizer.from_pretrained.return_value = "tockenizer"
    mock_automodel.from_pretrained.return_value = "model"
    model = ProjectNameModel(CNC_PATH)
    model.pretrained_model = None
    model.pretrained_tokenizer = "tockenizer"
    model.load_pretrained_llm(MODEL_NAME)
    mock_autotockenizer.from_pretrained.assert_not_called()
    mock_automodel.from_pretrained.assert_called_with(MODEL_NAME)
    assert model.pretrained_model == "model"
    assert model.pretrained_tokenizer == "tockenizer"
