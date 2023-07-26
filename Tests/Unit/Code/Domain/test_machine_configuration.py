import pandas as pd

from Code.Domain.Models.machine_configuration import MachineConfiguration


def test_machine_configuration_init():
    data = {'metric': [0.777],
            'post': ['post_example'],
            'machine': ['machine_example'],
            'file': ['file_example']}
    df = pd.DataFrame(data)
    post = MachineConfiguration(df)
    assert post.metric[0] == 0.78
    assert post.post[0] == 'post_example'
    assert post.machine[0] == 'machine_example'
    assert post.file[0] == 'file_example'


def test_machine_configuration_dict():
    data = {'metric': [0.777, 1],
            'post': ['post_example', 'example2'],
            'machine': ['machine_example', 'machine2'],
            'file': ['file_example', 'file2']}
    df = pd.DataFrame(data)
    post = MachineConfiguration(df)
    dictio = post.as_dict()
    assert dictio['metric'] == '{"0":0.78,"1":1.0}'
    assert dictio['post EXE File'] == '{"0":"post_example","1":"example2"}'
    assert dictio['machine'] == '{"0":"machine_example","1":"machine2"}'
