from unittest import mock

from Code.Application.smart_machine_config_model import SmartMachineConfigModel


@mock.patch("Code.Application.smart_machine_config_model.read_cnc_csv")
def test_init_smart_machine_model(mock_read_csv: mock):
    mock_read_csv.return_value = "test"
    model = SmartMachineConfigModel()
    assert model.cnc_df == "test"
    assert model.knowledge_kgrams == []
    assert model.knowledge_cnc_name == []
    mock_read_csv.assert_called_once()
