import time
from json import JSONDecodeError as JSONDecodeError2
from json.decoder import JSONDecodeError as JSONDecodeError1

from dynaconf import Dynaconf

from SRC.CORE.debug_utils import ERROR, get_local_ip
from SRC.CORE._CONSTANTS import project_root_dir, _IS_REDIS_CLOUD
from SRC.LIBRARIES.new_utils import timed_cache, check_env_true

REDIS_CLOUD_HOST = 'redis-12365.c328.europe-west3-1.gce.redns.redis-cloud.com'
REDIS_CLOUD_PORT = 12365
REDIS_CLOUD_USER = 'default'
REDIS_CLOUD_PASSWORD = 'oVEdaFt54BaS8eirMfGwRYLnIIjSWOLN'

REDIS_LOCAL_HOST = get_local_ip(print_out=False)
REDIS_LOCAL_PORT = 6379
REDIS_LOCAL_USER = None
REDIS_LOCAL_PASSWORD = None

REDIS_HOST = lambda: REDIS_CLOUD_HOST if check_env_true(_IS_REDIS_CLOUD, False) else REDIS_LOCAL_HOST
REDIS_PORT = lambda: REDIS_CLOUD_PORT if check_env_true(_IS_REDIS_CLOUD, False) else REDIS_LOCAL_PORT
REDIS_USER = lambda: REDIS_CLOUD_USER if check_env_true(_IS_REDIS_CLOUD, False) else REDIS_LOCAL_USER
REDIS_PASSWORD = lambda: REDIS_CLOUD_PASSWORD if check_env_true(_IS_REDIS_CLOUD, False) else REDIS_LOCAL_PASSWORD

SERVICE_ACCOUNT_FILE = f'{project_root_dir()}/secret-timing-381413-66a4b020e570.json'
MAILTRAP_CLIENT_TOKEN = '74b5682c0de95039bdc5c284f71d4992'
MAIL_SENDER = 'hello@demomailtrap.co'
MAIL_RECEIVER = 'andrii.korneichuk@uzhnu.edu.ua'

WEBDOCK_AKCRYPTOBUFF_HOST = 'akcryptobuff.vps.webdock.cloud'


@timed_cache(ttl=10)
def get_config(config_key, tryalls=5, default=None):
    from SRC.CORE.debug_utils import ERROR

    if tryalls == 0:
        return default

    try:
        config = Dynaconf(settings_files=[f"{project_root_dir()}/CONFIGS/webapp_configs.json"])
        
        if config.exists(config_key):
            return config[config_key]

        return default
    except (JSONDecodeError1, JSONDecodeError2):
        time.sleep(0.2)
        ERROR(f"RETRY GET CONFIG: {config_key} | TRYALLS REMAINS: {tryalls}")

        return get_config(config_key, tryalls-1, default)


__all__ = ['get_config']