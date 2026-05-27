from __future__ import annotations

import sys

from memori._cli import Cli
from memori._config import Config
from memori.provisioning import provision_memori, redact_dsn


class Manager:
    def __init__(self, config: Config):
        self.config = config

    def execute(self):
        cli = Cli(self.config)
        provider = _provider_from_argv(sys.argv[2:])
        if provider is None:
            self.usage()
            cli.newline()
            sys.exit(1)

        mem = provision_memori(provider=provider, build=True)
        result = mem.config.provision_result
        if result is None:
            raise RuntimeError("Provisioning completed without a result")

        cli.notice("Provisioned database is ready.")
        cli.notice(f"Provider: {result.provider}", 1)
        cli.notice(f"Family: {result.family}", 1)
        cli.notice(f"DSN: {redact_dsn(result.dsn)}", 1)
        if result.claim_url is not None:
            cli.notice(f"Claim URL: {result.claim_url}", 1)
        if result.expires_at is not None:
            cli.notice(f"Expires at: {result.expires_at}", 1)
        cli.newline()

        return self

    def usage(self):
        print("usage: python -m memori provision <provider>")
        print("       python -m memori provision --provider <provider>")


def _provider_from_argv(args: list[str]) -> str | None:
    if len(args) == 1 and not args[0].startswith("-"):
        return args[0]
    if len(args) == 2 and args[0] == "--provider":
        return args[1]
    return None
