r"""
 __  __                           _
|  \/  | ___ _ __ ___   ___  _ __(_)
| |\/| |/ _ \ '_ ` _ \ / _ \| '__| |
| |  | |  __/ | | | | | (_) | |  | |
|_|  |_|\___|_| |_| |_|\___/|_|  |_|
                  perfectam memoriam
                       memorilabs.ai
"""

from memori.storage._registry import Registry
from memori.storage.drivers.mysql._driver import Driver as MysqlDriver


@Registry.register_driver("oceanbase")
class Driver(MysqlDriver):
    """OceanBase storage driver (MySQL-compatible)."""
