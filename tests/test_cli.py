import subprocess
import sys

import pytest

from memori._cli import Cli
from memori._config import Config


@pytest.fixture
def mock_config():
    config = Config()
    config.version = "3.1.2"
    return config


def test_cli_banner_contains_key_elements(capsys, mock_config):
    """Test that banner output contains essential branding elements."""
    cli = Cli(config=mock_config)
    cli.banner()
    captured = capsys.readouterr()
    assert "Memori" in captured.out or "memori" in captured.out.lower()
    assert mock_config.version in captured.out
    assert "memorilabs.ai" in captured.out


def run_cli(*args):
    """
    Helper to run the CLI as a subprocess and capture results.
    """
    result = subprocess.run(
        [sys.executable, "-m", "memori", *args],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result


class TestCliEntrypoint:
    """End-to-end tests for CLI entrypoint behavior."""

    def test_cli_signup_missing_email_shows_error(self):
        """sign-up without email should show usage and exit with error."""
        result = run_cli("sign-up")
        assert result.returncode == 1
        # Verify it shows helpful error message
        assert "usage" in result.stdout.lower() or "email" in result.stdout.lower()

    def test_cli_output_contains_branding(self):
        """CLI output should contain Memori branding."""
        result = run_cli()
        output_lower = result.stdout.lower()
        assert "memori" in output_lower or "memorilabs" in result.stdout

    @pytest.mark.parametrize(
        "args",
        [
            [],
            ["--help"],
            ["-h"],
            ["help"],
        ],
    )
    def test_help_variations_show_all_commands(self, args):
        """Help output should show all commands for various help invocations."""
        result = run_cli(*args)
        assert result.returncode == 0
        assert "Params" in result.stdout
        assert "Option" in result.stdout
        assert "Description" in result.stdout
        assert "cockroachdb" in result.stdout
        assert "quota" in result.stdout
        assert "sign-up" in result.stdout
        assert "setup" in result.stdout
        assert "usage" in result.stdout

    def test_invalid_command_shows_help(self):
        """Invalid command should show help menu."""
        result = run_cli("AN_INVALID_COMMAND")
        assert "usage" in result.stdout.lower()
        assert "Option" in result.stdout
        assert "cockroachdb" in result.stdout
        assert "quota" in result.stdout

    def test_ascii_logo_displayed(self):
        """CLI should display the ASCII art Memori logo with tagline."""
        result = run_cli()
        assert "|  \\/  |" in result.stdout
        assert "| |\\/| |" in result.stdout
        assert "|_|  |_|" in result.stdout
        assert "perfectam memoriam" in result.stdout

    def test_cockroachdb_missing_subcommand_shows_usage(self):
        """cockroachdb without 'cluster' subcommand should show usage and exit 1"""
        result = run_cli("cockroachdb")

        assert result.returncode == 1
        assert "usage" in result.stdout.lower()
        assert "cluster" in result.stdout

    def test_cockroachdb_missing_action_shows_usage(self):
        """cockroachdb cluster without action should show usage and exit 1"""
        result = run_cli("cockroachdb", "cluster")

        assert result.returncode == 1
        assert "usage" in result.stdout.lower()
        assert (
            "start" in result.stdout
            or "claim" in result.stdout
            or "delete" in result.stdout
        )

    def test_cockroachdb_invalid_action_shows_usage(self):
        """cockroachdb cluster with invalid action should show usage and exit 1"""
        result = run_cli("cockroachdb", "cluster", "invalid-action")

        assert result.returncode == 1
        assert "usage" in result.stdout.lower()

    def test_cockroachdb_invalid_subcommand_shows_usage(self):
        """cockroachdb with invalid subcommand should show usage and exit 1"""
        result = run_cli("cockroachdb", "invalid-subcommand")

        assert result.returncode == 1
        assert "usage" in result.stdout.lower()

    @pytest.mark.parametrize(
        "action",
        ["start", "claim", "delete"],
    )
    def test_cockroachdb_valid_actions_recognized(self, action):
        """Valid cockroachdb cluster actions should be recognized (may fail for other reasons)"""
        result = run_cli("cockroachdb", "cluster", action)

        # Should NOT show usage error (command is recognized as valid)
        assert "usage: python -m memori cockroachdb cluster" not in result.stdout

        """ The command may fail for other reasons (network, missing cluster, etc.), but we just want to ensure the CLI recognizes it as a valid command """
