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


class Manager:
    def __init__(self, config: Config):
        self.config = config

    def execute(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "The setup command requires the optional embeddings dependency. "
                "Install it with `pip install 'memori[embeddings]'`."
            ) from exc

        cli = Cli(self.config)

        cli.notice("Installing model all-mpnet-base-v2")
        cli.notice("this may take a moment; output to follow:", 1)
        cli.notice("-----")

        SentenceTransformer("all-mpnet-base-v2")

        cli.notice("-----\n")

        return self
