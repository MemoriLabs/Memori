r"""
 __  __                           _
|  \/  | ___ _ __ ___   ___  _ __(_)
| |\/| |/ _ \ '_ ` _ \ / _ \| '__| |
| |  | |  __/ | | | | | (_) | |  | |
|_|  |_|\___|_| |_| |_|\___/|_|  |_|
                 perfectam memoriam
                      memorilabs.ai
"""

import warnings
from typing import TYPE_CHECKING, Any

from memori.llm._base import BaseProvider
from memori.llm.clients import Agno as AgnoMemoriClient
from memori.llm.clients import Anthropic as AnthropicMemoriClient
from memori.llm.clients import Google as GoogleMemoriClient
from memori.llm.clients import LangChain as LangChainMemoriClient
from memori.llm.clients import OpenAi as OpenAiMemoriClient
from memori.llm.clients import PydanticAi as PydanticAiMemoriClient
from memori.llm.clients import XAi as XAiMemoriClient

if TYPE_CHECKING:
    from memori import Memori


class Agno(BaseProvider):
    def register(
        self,
        openai_chat: Any | None = None,
        claude: Any | None = None,
        gemini: Any | None = None,
        xai: Any | None = None,
    ) -> "Memori":
        """Register Agno models.

        Deprecated: use `memori.llm.register(...)` with named parameters instead.
        """
        warnings.warn(
            "memori.agno.register() is deprecated. Use memori.llm.register(client) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self.client is None:
            self.client = AgnoMemoriClient(self.config).register(
                openai_chat=openai_chat,
                claude=claude,
                gemini=gemini,
                xai=xai,
            )

        return self.entity


class Anthropic(BaseProvider):
    def register(self, client: Any) -> "Memori":
        """Register an Anthropic client.

        Deprecated: use `memori.llm.register(client=...)` instead.
        """
        warnings.warn(
            "memori.anthropic.register() is deprecated. Use memori.llm.register(client) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self.client is None:
            self.client = AnthropicMemoriClient(self.config).register(client)

        return self.entity


class Google(BaseProvider):
    def register(self, client: Any) -> "Memori":
        """Register a Google client.

        Deprecated: use `memori.llm.register(client=...)` instead.
        """
        warnings.warn(
            "memori.google.register() is deprecated. Use memori.llm.register(client) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self.client is None:
            self.client = GoogleMemoriClient(self.config).register(client)

        return self.entity


class LangChain(BaseProvider):
    def register(
        self,
        chatbedrock: Any | None = None,
        chatgooglegenai: Any | None = None,
        chatopenai: Any | None = None,
        chatvertexai: Any | None = None,
    ) -> "Memori":
        """Register LangChain chat models.

        Deprecated: use `memori.llm.register(...)` with named parameters instead.
        """
        warnings.warn(
            "memori.langchain.register() is deprecated. Use memori.llm.register(client) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self.client is None:
            self.client = LangChainMemoriClient(self.config).register(
                chatbedrock=chatbedrock,
                chatgooglegenai=chatgooglegenai,
                chatopenai=chatopenai,
                chatvertexai=chatvertexai,
            )

        return self.entity


class OpenAi(BaseProvider):
    def register(self, client: Any, stream: bool = False) -> "Memori":
        """Register an OpenAI client.

        Deprecated: use `memori.llm.register(client=...)` instead.
        """
        warnings.warn(
            "memori.openai.register() is deprecated. Use memori.llm.register(client) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self.client is None:
            self.client = OpenAiMemoriClient(self.config).register(
                client, stream=stream
            )

        return self.entity


class PydanticAi(BaseProvider):
    def register(self, client: Any) -> "Memori":
        """Register a PydanticAI client.

        Deprecated: use `memori.llm.register(client=...)` instead.
        """
        warnings.warn(
            "memori.pydantic_ai.register() is deprecated. Use memori.llm.register(client) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self.client is None:
            self.client = PydanticAiMemoriClient(self.config).register(client)

        return self.entity


class XAi(BaseProvider):
    def register(self, client: Any, stream: bool = False) -> "Memori":
        """Register an XAI client.

        Deprecated: use `memori.llm.register(client=...)` instead.
        """
        warnings.warn(
            "memori.xai.register() is deprecated. Use memori.llm.register(client) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self.client is None:
            self.client = XAiMemoriClient(self.config).register(client, stream=stream)

        return self.entity
