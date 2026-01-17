import pytest

from memori._cli import Cli
from memori._config import Config


# A mock config object for instantiation
@pytest.fixture
def mock_config():
    config = Config()
    config.version = "3.1.2"
    return config


def test_cli_banner_prints_memori(capsys, mock_config):
    cli = Cli(config=mock_config)
    cli.banner()
    captured = capsys.readouterr()
    assert f"v{mock_config.version}" in captured.out
    assert "perfectam memoriam" in captured.out
    assert "memorilabs.ai" in captured.out


def test_cli_newline_prints_empty_line(capsys, mock_config):
    cli = Cli(config=mock_config)
    cli.newline()
    captured = capsys.readouterr()
    assert captured.out == "\n"


def test_cli_notice_prints_message_with_prefix(capsys, mock_config):
    cli = Cli(config=mock_config)
    cli.notice("Hello World")
    captured = capsys.readouterr()
    assert captured.out == "+ Hello World\n"


def test_cli_notice_prints_message_with_indent(capsys, mock_config):
    cli = Cli(config=mock_config)
    cli.notice("Indented message", ident=1)
    captured = capsys.readouterr()
    assert captured.out == "    Indented message\n"


def test_cli_print_prints_message(capsys, mock_config):
    cli = Cli(config=mock_config)
    cli.print("Raw message")
    captured = capsys.readouterr()
    assert captured.out == "Raw message\n"
