import os
from unittest import mock
from unittest.mock import MagicMock

from Code.Application.Services.model_services import predict_model_service, init_model_service

cwd = os.path.dirname(os.path.abspath(__file__))
FIXTURES_FOLDER = os.path.join(cwd, 'Fixtures/')
CNC_FILE = os.path.join(cwd, 'Fixtures/0000.GCD')
DATA = 'G02 X23.43 Y34.21'


@mock.patch('Code.Application.Services.model_services.cnc_path', FIXTURES_FOLDER)
@mock.patch('Code.Application.Services.model_services.read_cnc_csv')
@mock.patch('Code.Application.Services.model_services.pd')
@mock.patch('Code.Application.Services.model_services.load_ml_model')
@mock.patch('Code.Application.Services.model_services.model_fit')
def test_init_model_service(mock_fit: mock, mock_load_ml_model: mock, mock_pd: mock, mock_read: mock):
    mock_read.return_value = mock_pd.DataFrame
    mock_load_ml_model.return_value = None
    result = init_model_service()
    mock_fit.assert_called_once()
    mock_load_ml_model.assert_called()
    assert mock_load_ml_model.call_count == 2
    mock_read.assert_called_once()
    assert result


@mock.patch('Code.Application.Services.model_services.cnc_path', FIXTURES_FOLDER)
@mock.patch('Code.Application.Services.model_services.read_cnc_csv')
@mock.patch('Code.Application.Services.model_services.pd')
@mock.patch('Code.Application.Services.model_services.load_ml_model')
@mock.patch('Code.Application.Services.model_services.model_fit')
def test_noinit_model_service(mock_fit: mock, mock_load_ml_model: mock, mock_pd: mock, mock_read: mock):
    mock_read.return_value = mock_pd.DataFrame
    result = init_model_service()
    mock_fit.assert_not_called()
    mock_load_ml_model.assert_called_once()
    mock_read.assert_called_once()
    assert result


@mock.patch('Code.Application.Services.model_services.pd')
@mock.patch('Code.Application.Services.model_services.ProjectName')
@mock.patch('Code.Application.Services.model_services.pickle_model')
def test_predict_service(mock_pickle_model: mock, mock_post: mock, mock_pd):
    mock_pickle_model.predict_probea.return_value = mock_pd.DataFrame
    mock_post.return_value = MagicMock()
    predict_model_service(DATA)
    mock_post.assert_called_once()
    mock_post.return_value.as_dict.assert_called_once()
    mock_pickle_model.predict_probea.assert_called_once()
