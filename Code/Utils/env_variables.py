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
        self._host = os.getenv("HOST")
        self._port = os.getenv("PORT")

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
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port
