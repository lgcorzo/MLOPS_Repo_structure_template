import os
import pandas as pd

from unittest import mock
from Code.Application.smart_machine_config_algorithm import read_cnc_csv, split_text, read_file, intersection_count, \
    jacc_metric_multiset, remove_coordinate_numbers, clean_data_cncs
from Tests.Unit.Code.Application.test_smart_machine_config_evaluation import CNC_DF

cwd = os.path.dirname(os.path.abspath(__file__))

CNC_FOLDER_PATH = '../../../../Data/Raw/CNC'
CSV_PATH = os.path.join(cwd, 'Fixtures/test_cnc_ext.csv')
FIXTURES_FOLDER = os.path.join(cwd, 'Fixtures/')
TEST_CNC_FILE = '0000.GCD'


def test_split_text():
    input_str = 'asdfgh'
    output = split_text(input_str, 3)
    assert output == ['asd', 'sdf', 'dfg', 'fgh']


def test_read_file():
    cwd = os.path.dirname(os.path.abspath(__file__))
    path = 'Fixtures/test_cnc.txt'
    path_file = os.path.join(cwd, path)
    output = read_file(path_file)
    assert output == 'asd\nf\ngh'


def test_remove_coordinate_numbers():
    data = 'O0175 G03 X123.213 Y105432 Z-12.334 P-23'
    data_coord = remove_coordinate_numbers(data)
    assert data_coord == 'O0175 G03 X Y105432 Z P-23'


def test_clean_data_cncs():
    data_input = 'a10.25sd\ne6 g\r\nhi45.02 jk'
    data_output = clean_data_cncs(data_input)
    assert data_output == ['asd', 'sde', 'de6', 'e6 ', '6 g', ' gh', 'ghi', 'hi ', 'i j', ' jk']


@mock.patch('Code.Application.smart_machine_config_algorithm.csv_path', CSV_PATH)
def test_read_cnc_csv():
    test_df = read_cnc_csv()
    pd.testing.assert_frame_equal(test_df, CNC_DF)


def test_intersection_count():
    a = ['a', 'a', 'b']
    b = ['a', 'a', 'b', 'c']
    output = intersection_count(a, b)
    assert output == 3


def test_jacc_metric_multiset():
    a = ['a', 'a', 'b']
    b = ['a', 'a', 'b', 'c']
    output = jacc_metric_multiset(a, b)
    assert output == 2 * 3 / 7


def test_jacc_metric_multiset_list():
    a = []
    b = ['a']
    output = jacc_metric_multiset(a, b)
    assert output == 0
