import os
import base64
import json

from unittest import mock
from Code.Controller.app import app

CNC_FOLDER_PATH = '../../../../Data/Results/CNC/'
TEST_CNC_FILE = 'Fixtures/0000.GCD'
MACHINE_CONFIG_ENDPOINT = 'machine-configuration'

cwd = os.path.dirname(os.path.abspath(__file__))
fixtures_folder = os.path.join(cwd, 'Fixtures/')
files_path = os.path.join(cwd, CNC_FOLDER_PATH)
cnc_file_path = os.path.join(cwd, TEST_CNC_FILE)


@mock.patch('Code.Application.Services.model_services.cnc_path', files_path)
@mock.patch('Code.Controller.app.predict_model_service')
def test_predict(mock_predict_service: mock):
    mock_predict_service.return_value = {'file': '0000.GCD'}
    response = app.test_client().post('/services/' + MACHINE_CONFIG_ENDPOINT, data={
        '': open(cnc_file_path, 'rb')
    })
    res = json.loads(response.data.decode('utf-8'))
    assert response.status_code == 200
    mock_predict_service.assert_called()
    assert res == {'file': '0000.GCD'}


@mock.patch('Code.Application.Services.model_services.cnc_path', files_path)
@mock.patch('Code.Controller.app.predict_model_service')
def test_dash_postprocessor_post(mock_predict_service: mock):
    mock_predict_service.return_value = {'file': '0000.GCD'}
    content = open(cnc_file_path, 'r', encoding='ISO-8859-1').read()
    coded = content.encode("ascii")
    file_content = base64.b64encode(coded)
    response = app.test_client().post('/services/dash-machine-configuration', data={
        '': file_content
    })
    res = json.loads(response.data.decode('utf-8'))
    assert response.status_code == 200
    mock_predict_service.assert_called()
    assert res == {'file': '0000.GCD'}


def test_machine_configuration():
    response = app.test_client().get('/services/machine-configuration')
    res = json.loads(response.data.decode('utf-8'))
    assert response.status_code == 200
    assert res['status'] == 'alive!'


@mock.patch('Code.Controller.app.feedback_service')
def test_send_feedback(mock_feedback: mock):
    exam_json = {"id": "agadsghdgshgd", "validation": True, "comments": "test successful"}
    response = app.test_client().post('/services/feedback', json=exam_json)
    assert response.status_code == 200
    mock_feedback.assert_called()


@mock.patch('Code.Controller.app.predict_model_service')
def test_prediction(mock_predict: mock):
    mock_predict.return_value = {'file': '0000.GCD'}
    response = app.test_client().post('/services/' + MACHINE_CONFIG_ENDPOINT, data={
        '': open(cnc_file_path, 'rb')
    })
    assert response.status_code == 200
    res = json.loads(response.data.decode('utf-8'))
    mock_predict.assert_called()
    assert res == {'file': '0000.GCD'}
