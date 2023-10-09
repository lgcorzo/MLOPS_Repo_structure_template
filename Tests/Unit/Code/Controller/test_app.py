import os
import base64
import json

from unittest import mock
from fastapi.testclient import TestClient
import pytest
from Code.Controller.app import app


CNC_FOLDER_PATH = '../../../../Data/Results/CNC/'
TEST_CNC_FILE = 'Fixtures/0000.GCD'
SERVICE_ENDPOINT = 'model-predict'
cwd = os.path.dirname(os.path.abspath(__file__))
fixtures_folder = os.path.join(cwd, 'Fixtures/')
files_path = os.path.join(cwd, CNC_FOLDER_PATH)
cnc_file_path = os.path.join(cwd, TEST_CNC_FILE)


@pytest.fixture
def client():
    return TestClient(app)


@mock.patch('Code.Application.Services.model_services.cnc_path', files_path)
@mock.patch('Code.Controller.app.predict_model_service')
def test_predict(mock_predict_service: mock, client: TestClient) -> None:
    mock_predict_service.return_value = {'file_data': '0000.GCD'}
    response = client.post('/services/' + SERVICE_ENDPOINT, files={
        "cnc_file": str(open(cnc_file_path, 'rb'))})
    res = json.loads(response.content.decode('utf-8'))
    assert response.status_code == 200
    mock_predict_service.assert_called()
    assert res == {'file_data': '0000.GCD'}


@mock.patch('Code.Application.Services.model_services.cnc_path', files_path)
@mock.patch('Code.Controller.app.predict_model_service')
@mock.patch('Code.Controller.app.init_model_service')
def test_dash_postprocessor_post(mock_init_model_service: mock, mock_predict_service: mock, client: TestClient):
    mock_predict_service.return_value = {'file_data': '0000.GCD'}
    content = open(cnc_file_path, 'r', encoding='ISO-8859-1').read()
    coded = content.encode("ascii")
    file_content = base64.b64encode(coded)
    file_string = str(file_content, 'ISO-8859-1')
    response = client.post('/services/dash-model-predict', data={
        "file_data": file_string
    })
    res = json.loads(response.content.decode('utf-8'))
    assert response.status_code == 200
    mock_predict_service.assert_called()
    assert res == {'file_data': '0000.GCD'}


@mock.patch('Code.Controller.app.init_model_service')
def test_service_is_alive(mock_init_model_service: mock, client: TestClient):
    response = client.get('/services/is-alive')
    res = json.loads(response.content.decode('utf-8'))
    assert response.status_code == 200
    assert res['status'] == 'alive!'


@mock.patch('Code.Controller.app.feedback_service')
@mock.patch('Code.Controller.app.init_model_service')
def test_send_feedback(mock_init_model_service: mock, mock_feedback: mock, client: TestClient):
    exam_json = {"id": "agadsghdgshgd", "validation": True, "comments": "test successful"}
    response = client.post('/services/feedback', json=exam_json)
    assert response.status_code == 200
    mock_feedback.assert_called()


@mock.patch('Code.Controller.app.predict_model_service')
@mock.patch('Code.Controller.app.init_model_service')
def test_prediction(mock_init_model_service: mock, mock_predict: mock, client: TestClient):
    mock_predict.return_value = {'file': '0000.GCD'}
    response = client.post('/services/' + SERVICE_ENDPOINT, files={
        'cnc_file': open(cnc_file_path, 'rb')
    })
    assert response.status_code == 200
    res = json.loads(response.content.decode('utf-8'))
    mock_predict.assert_called()
    assert res == {'file_data': '0000.GCD'}
