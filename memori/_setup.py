r"""
 __  __                           _
|  \/  | ___ _ __ ___   ___  _ __(_)
| |\/| |/ _ \ '_ ` _ \ / _ \| '__| |
| |  | |  __/ | | | | | (_) | |  | |
|_|  |_|\___|_| |_| |_|\___/|_|  |_|
                  perfectam memoriam
                       memorilabs.ai
"""

from memori._cli import Cli
from memori._config import Config
from memori.native import embed_texts


class Manager:
    def __init__(self, config: Config):
        self.config = config

    def execute(self):
        cli = Cli(self.config)
        model = self.config.embeddings.model

        cli.notice(f"Installing embedding model {model}")
        cli.notice("this may take a moment; output to follow:", 1)
        cli.notice("-----")

        embed_texts(["memori setup"], model=model)

        cli.notice("-----\n")

        return self
