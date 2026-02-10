from dataclasses import dataclass


@dataclass
class ConversationMessage:
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}
