from unittest import mock
from Code.Utils.env_variables import Env
from Code.Utils.env_variables import Singleton


@mock.patch('Code.Utils.env_variables.os.getenv')
@mock.patch('Code.Utils.env_variables.load_dotenv')
def test_env_properties(mock_load_env: mock, mock_getenv: mock):
    mock_getenv.side_effect = [
        '123456',
        '78910',
        '111213',
        '141516',
        'subscription_alias',
        'product_alias',
        'resource_group',
        'environment_alias',
        'mlw_workspace_name',
        'localhost',
        800,
        'localhost',
        800
    ]

    env = Env()
    mock_load_env.assert_called_once()
    assert env.subscription_id == '123456'
    assert env.tenant_id == '78910'
    assert env.client_id == '111213'
    assert env.client_secret == '141516'
    assert env.subscription_alias == 'subscription_alias'
    assert env.product_alias == 'product_alias'
    assert env.resource_group == 'resource_group'
    assert env.environment_alias == 'environment_alias'
    assert env.mlw_workspace_name == 'mlw_workspace_name'
    assert env.be_host == 'localhost'
    assert env.be_port == 800
    assert env.fe_host == 'localhost'
    assert env.fe_port == 800


def test_singleton_instance():
    instance1 = Singleton()
    instance2 = Singleton()
    assert instance1 is instance2
