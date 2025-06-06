from enum import Enum
from dataclasses import dataclass


class AppEnvTypes(Enum):
    prod: str = "prod"
    dev: str = "dev"


@dataclass
class BaseAppSettings:
    app_env: AppEnvTypes = AppEnvTypes.dev
    # app_env: AppEnvTypes = AppEnvTypes.prod
