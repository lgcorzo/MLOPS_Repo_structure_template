import os
import pandas as pd
import pytest
from unittest import mock
from unittest.mock import MagicMock
from Code.Application.project_name_evaluation import model_fit, model_train, get_evaluation_results, main,\
    train_test_split_parts, load_cncs, create_train_test_dict

cwd = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = 'test_cnc_ext.csv'
FIXTURES_PATH = os.path.join(cwd, 'Fixtures', 'CNC')
CNC_PATH = os.path.join(cwd, 'Fixtures', 'CNC', CSV_FILE)
POST = 'pst.exe'
SEED = 'pst.se1'
PART = 'DXFPAR0105'
DATA_DICT = {'WrkRef': ['Machine', 'Machine', 'Machine', 'Machine'],
             'CNC': ['1913', '1914', '1915', '1916'],
             'DIS_PsfFile': [POST, POST, POST, POST],
             'DIS_CfgFile': [SEED, SEED, SEED, SEED],
             'PrdRefDst': ['Pieza_prueba_Maxwell', 'Pieza_prueba_Maxwell_2',
                           'DXFPART7', PART],
             'Extension': ['.CNC', '.CNC', '.CNC', '.CNC']}
CNC_DF = pd.DataFrame(DATA_DICT)


def test_train_test_split_parts() -> None:
    train_df, test_df = train_test_split_parts(CNC_DF, PART)
    pd.testing.assert_frame_equal(test_df, CNC_DF[CNC_DF['PrdRefDst'] == PART])
    pd.testing.assert_frame_equal(train_df, CNC_DF[CNC_DF['PrdRefDst'] != PART])


@mock.patch('Code.Application.project_name_evaluation.read_file')
def test_load_cnc_kgram(mock_read_cnc):
    result_dict = load_cncs(FIXTURES_PATH, CNC_DF)
    assert mock_read_cnc.call_count == len(CNC_DF)
    assert list(result_dict.keys()) == ['1913.CNC', '1914.CNC', '1915.CNC', '1916.CNC']


def test_create_train_test_dict():
    kgram_list = ['abc', 'bcd']
    input_dict = {'1913.CNC': kgram_list,
                  '1914.CNC': kgram_list,
                  '1915.CNC': kgram_list,
                  '1916.CNC': kgram_list}
    input_train_dict, input_test_dict = create_train_test_dict(input_dict, CNC_DF, PART)
    assert list(input_train_dict.keys()) == ['1913.CNC', '1914.CNC', '1915.CNC']
    assert list(input_test_dict.keys()) == ['1916.CNC']
    assert list(input_train_dict.values()) == [kgram_list, kgram_list, kgram_list]
    assert list(input_test_dict.values()) == [kgram_list]


@pytest.mark.skip(reason="testing")
@mock.patch('Code.Application.project_name_evaluation.print')
@mock.patch('Code.Application.project_name_evaluation.load_cncs')
@mock.patch('Code.Application.project_name_evaluation.pickle')
def test_model_fit(mock_pickle: mock, mock_load_cnc: mock, mock_print: mock):
    mock_load_cnc.return_value = {'file1.cnc': ['abc', 'bcd'],
                                  'file2.cnc': ['efg', 'fgh']}
    model_fit(FIXTURES_PATH, CNC_DF, CNC_PATH)
    mock_pickle.dump.assert_called_once()
    mock_print.assert_called()


@mock.patch('Code.Application.project_name_algorithm.RAW_PATH', FIXTURES_PATH)
@mock.patch('Code.Application.project_name_algorithm.CSV_FILE', CSV_FILE)
@mock.patch('Code.Application.project_name_evaluation.mlflow')
def test_model_train(mock_mlflow: mock) -> None:
    score = model_train(FIXTURES_PATH, PART, CNC_DF, CNC_PATH)
    assert score == 0.5
    mock_mlflow.log_metric.assert_called_with("accuracy", 0.5)


@mock.patch('Code.Application.project_name_evaluation.logging')
@mock.patch('Code.Application.project_name_evaluation.model_train')
def test_get_evaluation_results(mock_train: mock, mock_logging: mock) -> None:
    data_test = {'file': ['test1', 'test2'],
                 'post': ['post.exe', 'post.exe'],
                 'PrdRefDst': ['part1', 'part2']}
    input_df = pd.DataFrame(data_test)
    mock_train.side_effect = [1, 0.9]
    deploy = get_evaluation_results(FIXTURES_PATH, input_df)
    mock_train.assert_called()
    mock_logging.info.assert_called()
    assert deploy


@mock.patch('Code.Application.project_name_evaluation.read_cnc_csv')
@mock.patch('Code.Application.project_name_evaluation.get_evaluation_results')
@mock.patch('Code.Application.project_name_evaluation.cnc_path', FIXTURES_PATH)
def test_main(mock_evaluate: mock, mock_read: mock):
    main()
    mock_read.assert_called_once()
    mock_evaluate.assert_called_once()
