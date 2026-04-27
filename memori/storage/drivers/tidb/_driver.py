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
from memori.storage.migrations._tidb import migrations


@Registry.register_driver("tidb")
class Driver(MysqlDriver):
    """TiDB storage driver (MySQL-compatible distributed SQL)."""

    migrations = migrations
    requires_rollback_on_error = False
