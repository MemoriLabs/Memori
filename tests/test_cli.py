import subprocess
import sys
from unittest import mock

import pytest

from memori.__main__ import main
from memori._cli import Cli
from memori._config import Config


@pytest.fixture
def mock_config():
    config = Config()
    config.version = "3.1.2"
    return config


def test_cli_banner_contains_key_elements(capsys, mock_config):
    cli = Cli(config=mock_config)
    cli.banner()
    captured = capsys.readouterr()
    assert "Memori" in captured.out or "memori" in captured.out.lower()
    assert mock_config.version in captured.out
    assert "memorilabs.ai" in captured.out


@pytest.mark.integration
def test_entrypoint_smoke_run():
    result = subprocess.run(
        [sys.executable, "-m", "memori", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "usage" in result.stdout.lower()


class TestCliEntrypoint:
    def run_main_with_args(self, args):
        with mock.patch.object(sys, "argv", ["memori"] + args):
            try:
                main()
            except SystemExit as e:
                return e.code
            return 0

    def test_cli_signup_missing_email_shows_error(self, capsys):
        exit_code = self.run_main_with_args(["sign-up"])
        assert exit_code != 0
        captured = capsys.readouterr()
        output = captured.out.lower()
        assert "usage" in output or "email" in output

    def test_cli_no_args_shows_branding(self, capsys):
        self.run_main_with_args([])
        captured = capsys.readouterr()
        output = captured.out.lower()
        assert "memori" in output or "memorilabs" in output
        assert "usage" in output

    @pytest.mark.parametrize(
        "args",
        [
            ["--help"],
            ["-h"],
            ["help"],
        ],
    )
    def test_help_variations_show_all_commands(self, args, capsys):
        self.run_main_with_args(args)
        captured = capsys.readouterr()
        output = captured.out.lower()
        assert "usage" in output
        assert "cockroachdb" in output
        assert "quota" in output
        assert "sign-up" in output
        assert "setup" in output

    def test_invalid_command_shows_help(self, capsys):
        self.run_main_with_args(["invalid-command"])
        captured = capsys.readouterr()
        output = captured.out.lower()
        assert "usage" in output
        assert "cockroachdb" in output
        assert "sign-up" in output

    def test_branding_displayed(self, capsys):
        self.run_main_with_args([])
        captured = capsys.readouterr()
        output = captured.out.lower()
        assert "memori" in output
        assert "memorilabs.ai" in output

    def test_cockroachdb_missing_subcommand_shows_usage(self, capsys):
        exit_code = self.run_main_with_args(["cockroachdb"])
        assert exit_code != 0
        captured = capsys.readouterr()
        assert "usage" in captured.out.lower()

    @mock.patch("memori.__main__.CockroachDBClusterManager")
    def test_cockroachdb_cluster_start_dispatches_correctly(
        self, mock_manager_cls, capsys
    ):
        mock_instance = mock_manager_cls.return_value
        mock_instance.execute.return_value = None
        exit_code = self.run_main_with_args(["cockroachdb", "cluster", "start"])
        assert exit_code in (0, None)
        mock_manager_cls.assert_called()
        mock_instance.execute.assert_called()
        captured = capsys.readouterr()
        assert "usage: python -m memori cockroachdb cluster" not in captured.out
