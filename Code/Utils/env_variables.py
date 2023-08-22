import os

from dotenv import load_dotenv


class Singleton(object):
    _instances = {}

    def __new__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__new__(cls, *args, **kwargs)
        return cls._instances[cls]


class Env(Singleton):

    def __init__(self) -> None:
        load_dotenv()

        self._arm_subscription_id = os.getenv("ARM_SUBSCRIPTION_ID")
        self._arm_tenant_id = os.getenv("ARM_CLIENT_ID")
        self._arm_client_id = os.getenv("ARM_TENANT_ID")
        self._arm_client_secret = os.getenv("ARM_CLIENT_SECRET")
        self._subscription_alias = os.getenv("SUBSCRIPTION_ALIAS")
        self._product_alias = os.getenv("PRODUCT_ALIAS")
        self._resource_group = os.getenv("RESOURCE_GROUP")
        self._environment_alias = os.getenv("ENVIRONMENT_ALIAS")
        self._mlw_workspace_name = os.getenv("MLW_WORKSPACE_NAME")
        self._be_host = os.getenv("BE_HOST")
        self._be_port = os.getenv("BE_PORT")
        self._fe_host = os.getenv("FE_HOST")
        self._fe_port = os.getenv("FE_PORT")
        self._experiment_name = os.getenv("EXPERIMENT_NAME")
        self._remote_server_uri = os.getenv("MLFLOW_REMOTE_SERVER_URI")
        self._run_name = os.getenv("RUN_NAME")
        self._registered_model_name = os.getenv("REGISTERED_MODEL_NAME")

    @property
    def subscription_id(self) -> str:
        return self._arm_subscription_id

    @property
    def tenant_id(self) -> str:
        return self._arm_tenant_id

    @property
    def client_id(self) -> str:
        return self._arm_client_id

    @property
    def client_secret(self) -> str:
        return self._arm_client_secret

    @property
    def subscription_alias(self) -> str:
        return self._subscription_alias

    @property
    def product_alias(self) -> str:
        return self._product_alias

    @property
    def resource_group(self) -> str:
        return self._resource_group

    @property
    def environment_alias(self) -> str:
        return self._environment_alias

    @property
    def mlw_workspace_name(self) -> str:
        return self._mlw_workspace_name

    @property
    def be_host(self) -> str:
        return self._be_host

    @property
    def be_port(self) -> int:
        return self._be_port

    @property
    def fe_host(self) -> str:
        return self._fe_host

    @property
    def fe_port(self) -> int:
        return self._fe_port

    @property
    def experiment_name(self) -> int:
        return self._experiment_name

    @property
    def remote_server_uri(self) -> int:
        return self._remote_server_uri

    @property
    def run_name(self) -> int:
        return self._run_name

    @property
    def registered_model_name(self) -> int:
        return self._registered_model_name
