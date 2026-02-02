r"""
 _  __                           _
|  \/  | ___ _ __ ___   ___  _ __(_)
| |\/| |/ _ \ '_ ` _ \ / _ \| '__| |
| |  | |  __/ | | | | | (_) | |  | |
|_|  |_|\___|_| |_| |_|\___/|_|  |_|
                 perfectam memoriam
                      memorilabs.ai
"""

import warnings
from importlib.metadata import PackageNotFoundError, distribution


class QuotaExceededError(Exception):
    def __init__(
        self,
        message=(
            "your IP address is over quota; register for an API key now: "
            + "https://app.memorilabs.ai/signup"
        ),
    ):
        self.message = message
        super().__init__(self.message)


class MemoriApiError(Exception):
    pass


class MemoriApiClientError(MemoriApiError):
    def __init__(
        self,
        status_code: int,
        message: str | None = None,
        details: object | None = None,
    ):
        self.status_code = status_code
        self.details = details
        super().__init__(
            message or f"Memori API request failed with status {status_code}"
        )


class MemoriApiValidationError(MemoriApiClientError):
    pass


class MemoriApiRequestRejectedError(MemoriApiClientError):
    pass


class MissingMemoriApiKeyError(RuntimeError):
    """Raised when hosted mode is used without a MEMORI_API_KEY."""

    def __init__(self, env_var: str = "MEMORI_API_KEY"):
        self.env_var = env_var
        super().__init__(
            f"A {env_var} is required to use the Memori hosted API. Sign up at https://app.memorilabs.ai/signup"
        )


class UnsupportedLLMProviderError(RuntimeError):
    """Raised when an unsupported LLM provider is used."""

    def __init__(self, provider: str):
        super().__init__(
            f"Unsupported LLM provider: {provider}. Please see the documentation for supported providers: https://memorilabs.ai/docs/features/llm"
        )


class UnsupportedDatabaseError(RuntimeError):
    """Raised when an unsupported database is used."""

    def __init__(self, database: str | None = None):
        msg = (
            "Unsupported database."
            if database is None
            else f"Unsupported database: {database}."
        )
        super().__init__(
            f"{msg} Please see the documentation for supported databases: https://memorilabs.ai/docs/features/databases"
        )


class MemoriLegacyPackageWarning(UserWarning):
    """Warning emitted when the legacy `memorisdk` package is installed."""


def warn_if_legacy_memorisdk_installed() -> None:
    try:
        distribution("memorisdk")
    except PackageNotFoundError:
        return

    warnings.warn(
        "You have Memori installed under the legacy package name 'memorisdk'. "
        "That name is deprecated and will stop receiving updates. "
        "Please switch to 'memori':\n\n"
        "    pip uninstall memorisdk\n"
        "    pip install memori\n",
        MemoriLegacyPackageWarning,
        stacklevel=3,
    )
