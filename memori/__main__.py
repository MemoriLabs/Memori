r"""
 __  __                           _
|  \/  | ___ _ __ ___   ___  _ __(_)
| |\/| |/ _ \ '_ ` _ \ / _ \| '__| |
| |  | |  __/ | | | | | (_) | |  | |
|_|  |_|\___|_| |_| |_|\___/|_|  |_|
                  perfectam memoriam
                       memorilabs.ai
"""

import os
import sys
from pathlib import Path
from typing import Any

from memori._cli import Cli
from memori._config import Config
from memori._setup import Manager as SetupManager
from memori.api._quota import Manager as ApiQuotaManager
from memori.api._sign_up import Manager as ApiSignUpManager
from memori.storage.cockroachdb._cluster_manager import (
    ClusterManager as CockroachDBClusterManager,
)


def _load_env_file() -> None:
    env_path = Path.cwd() / ".env"
    if not env_path.is_file():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if stripped.startswith("export "):
            stripped = stripped.removeprefix("export ").lstrip()

        key, separator, value = stripped.partition("=")
        key = key.strip()
        if not separator or not key or key in os.environ:
            continue

        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].strip()

        os.environ[key] = value


def main():
    _load_env_file()

    cli = Cli(Config())
    cli.banner()

    options: dict[str, dict[str, Any]] = {
        "cockroachdb": {
            "description": "Manager a CockroachDB cluster",
            "params": ["cluster", "<start | claim | delete>"],
            "obj": CockroachDBClusterManager,
        },
        "quota": {
            "description": "Check your quota",
            "params": [],
            "obj": ApiQuotaManager,
        },
        "setup": {
            "description": "Execute suggested setup steps",
            "params": [],
            "obj": SetupManager,
        },
        "sign-up": {
            "description": "Sign up for an API key",
            "params": ["<email_address>"],
            "obj": ApiSignUpManager,
        },
    }

    if len(sys.argv) <= 1 or sys.argv[1] not in options:
        cli.print("{:<15}{:<45}{:<6}".format("Option", "Description", "Params"))
        cli.print("{:<15}{:<45}{:<6}".format("------", "-----------", "------"))

        for key, value in options.items():
            params = value["params"]
            cli.print(
                "{:<15}{:<45}{:>6}".format(
                    key, value["description"], "Y" if len(params) > 0 else "N"
                )
            )

        cli.print("\nusage: python -m memori <option> [params]\n")
    else:
        option = options[sys.argv[1]]
        params = option["params"]
        obj_cls = option["obj"]
        if len(params) > 0:
            if len(sys.argv) != 2 + len(params):
                obj_cls(Config()).usage()
                cli.newline()
                sys.exit(1)

        obj_cls(Config()).execute()


if __name__ == "__main__":
    main()
