import json
import os

from unittest import mock

from Code.Application.Services.model_services import predict_model_service, fit_model_service
from Code.Application.project_name_algorithm import read_file

cwd = os.path.dirname(os.path.abspath(__file__))
CNC_PATH = os.path.join(cwd, 'Fixtures')
CNC_FILE_PATH = os.path.join(CNC_PATH, '2.NC')
PKL_PATH = os.path.join(CNC_PATH, 'project_name_model.pkl')
CSV_PATH = os.path.join(CNC_PATH, 'test_list_cnc.csv')


@mock.patch('Code.Application.Services.model_services.cnc_path', CNC_PATH)
@mock.patch('Code.Application.project_name_evaluation.pkl_path', PKL_PATH)
@mock.patch('Code.Application.project_name_algorithm.csv_path', CSV_PATH)
def test_fit_model_service():
    assert fit_model_service()
    assert os.path.exists(PKL_PATH)


@mock.patch('Code.Application.project_name_evaluation.cnc_path', CNC_PATH)
@mock.patch('Code.Application.project_name_evaluation.pkl_path', PKL_PATH)
def test_predict_service():
    file_content = read_file(CNC_FILE_PATH)
    result = predict_model_service(file_content)
    metric = json.loads(result['metric'])
    post = json.loads(result['post EXE File'])
    assert metric['0'] == 1
    assert post['0'] == 'pstama02.exe'
    os.remove(PKL_PATH)
