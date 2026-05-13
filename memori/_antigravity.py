from memori._cli import Cli
from memori._config import Config


class Manager:
    def __init__(self, config: Config):
        self.config = config
        self.cli = Cli(config)

    def usage(self):
        pass

    def execute(self):
        self.cli.notice("Activating antigravity...")
        self.cli.newline()
        self.cli.print("  .   *   .  *  .  *   .   .")
        self.cli.print(" *  .  \   /  .   *   .  *")
        self.cli.print("  .  *  - O -  *   .   .  ")
        self.cli.print(" .   .  /   \  .   *   .  ")
        self.cli.print("  *   .   *   .   .   *   ")
        self.cli.newline()
        self.cli.notice("Whoa, I'm flying! 🎈")
        
        try:
            import antigravity
        except Exception:
            pass
