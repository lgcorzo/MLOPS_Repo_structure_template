import os
import pandas as pd
import pytest

from pandas import DataFrame
from unittest import mock

from Code.Application.smart_machine_config_algorithm import split_text, read_file, NUM_GRAM
from Code.Application.smart_machine_config_model import SmartMachineConfigModel, get_file_column_from_probea_results, \
    app_jacc_metric_multiset

DIN_FILE = ['M25', 'M25', 'M25',
            'M29', 'M29', 'M29',
            'M14', 'M14', 'M14',
            'M15', 'M15', 'M15']

cwd = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = 'test_list_cnc.csv'
csv_path = os.path.join(cwd, 'Fixtures/CNC/test_list_cnc.csv')
fixtures_folder = os.path.join(cwd, 'Fixtures')
fixtures_cnc_folder = os.path.join(cwd, 'Fixtures/CNC')
test_cnc_file = os.path.join(cwd, 'Fixtures/CNC/0000.CNC')


@pytest.fixture
@mock.patch('Code.Application.smart_machine_config_algorithm.RAW_PATH', fixtures_folder)
@mock.patch('Code.Application.smart_machine_config_algorithm.CSV_FILE', CSV_FILE)
def fixture_init_model():
    return SmartMachineConfigModel(csv_path)


@pytest.fixture
def fixture_fit_model(fixture_init_model):
    model = fixture_init_model
    path_files = fixtures_cnc_folder
    kwnoledge_kgrams = [split_text(read_file(os.path.join(path_files, cnc_name)), NUM_GRAM)
                        for cnc_name in os.listdir(path_files)]
    knowledge_cnc_name = os.listdir(path_files)
    model.fit(kwnoledge_kgrams, knowledge_cnc_name)
    return model


def test_model_fit(fixture_init_model):
    model = fixture_init_model
    knowledge_kgrams = [['ABC1', 'CDE1', 'FGH1'], ['ABC2', 'CDE2', 'FGH2']]
    knowledge_cnc_name = ['cluster_1', 'cluster_2']

    model.fit(knowledge_kgrams, knowledge_cnc_name)

    assert model.knowledge_kgrams == knowledge_kgrams
    assert model.knowledge_cnc_name == knowledge_cnc_name


def test_model_predict_probea_1(fixture_fit_model):
    model = fixture_fit_model
    data = {'metric': [1.0],
            'post': ['pst0.exe'],
            'machine': ['Machine'],
            'file': ['0000.CNC']}
    expected_result = pd.DataFrame(data)
    split_comp = split_text(read_file(test_cnc_file), NUM_GRAM)
    data = {'cnc_db': split_comp}
    data_serie = pd.Series(data)
    result_out = model.predict_probea(data_serie, 1)
    assert isinstance(result_out, DataFrame)
    pd.testing.assert_frame_equal(result_out, expected_result)


def test_model_predict_probea_2(fixture_fit_model):
    model = fixture_fit_model
    data = {'metric': [1.0, 0.9175475687103594],
            'post': ['pst0.exe', 'pst1.exe'],
            'machine': ['Machine', 'Machine'],
            'file': ['0000.CNC', '0000.DAT']}
    expected_result = pd.DataFrame(data)
    split_comp = split_text(read_file(test_cnc_file), NUM_GRAM)
    data = {'cnc_db': split_comp}
    data_serie = pd.Series(data)
    result_out = model.predict_probea(data_serie, 2)
    assert isinstance(result_out, DataFrame)
    pd.testing.assert_frame_equal(result_out, expected_result)


def test_model_score(fixture_fit_model):
    model = fixture_fit_model
    knowledge_kgrams = [DIN_FILE, DIN_FILE]
    knowledge_cnc_name = ['0000.DIN', '0000.DAT']
    score = model.score(knowledge_kgrams, knowledge_cnc_name)
    assert score == 0.5


def test_model_predict(fixture_fit_model):
    model = fixture_fit_model
    result_out = model.predict([DIN_FILE])
    assert result_out[0] == '0000.DIN'


def test_get_file_column_from_probea_results() -> None:
    org_df = pd.DataFrame({
        'cnc_db': ['matref_1', 'matref_1', 'matref_2', 'matref_3', 'matref_4'],
        'file': ['matref_1', 'matref_1', 'matref_2', 'matref_3', 'matref_4'],
    })
    expexted_df = pd.DataFrame({
        'file': ['matref_1', 'matref_1', 'matref_2', 'matref_3', 'matref_4'],
    })
    result_out = org_df.apply(get_file_column_from_probea_results, axis=1)
    pd.testing.assert_series_equal(result_out, expexted_df['file'], check_names=False)


def test_app_jacc_metric_multiset() -> None:

    split_comp = [split_text(read_file(test_cnc_file), NUM_GRAM),
                  split_text(read_file(test_cnc_file), NUM_GRAM)]
    org_df = pd.DataFrame()
    org_df['cnc_db'] = split_comp
    org_df['metric'] = 1.0
    data = {'cnc_db': split_text(read_file(test_cnc_file), NUM_GRAM)}
    cnc_df = pd.Series(data)
    expected_result = pd.Series([1.0, 1.0])
    result_out = org_df.apply(app_jacc_metric_multiset, cnc_comp=cnc_df, axis=1)
    pd.testing.assert_series_equal(result_out, expected_result)
