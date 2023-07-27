from unittest import mock
import os
from Code.Application.project_name_model import ProjectNameModel


cwd = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = 'test_cnc_ext.csv'
FIXTURES_PATH = os.path.join(cwd, 'Fixtures', 'CNC')
CNC_PATH = os.path.join(cwd, 'Fixtures', 'CNC', CSV_FILE)


@mock.patch("Code.Application.project_name_model.read_cnc_csv")
def test_init_project_name_model(mock_read_csv: mock):
    mock_read_csv.return_value = "test"
    model = ProjectNameModel(CNC_PATH)
    assert model.cnc_df == "test"
    assert model.knowledge_kgrams == []
    assert model.knowledge_cnc_name == []
    mock_read_csv.assert_called_once()
