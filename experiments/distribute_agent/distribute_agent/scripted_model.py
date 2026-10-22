"""A deterministic, offline SmolAgents model for wiring tests.

This is a real ``Model`` implementation: CodeAgent parses and executes its
output exactly as it would for a hosted model. Tests use it to prove that the
skill, package metadata and sandbox capability are wired together. It says
nothing about whether a real LLM would follow the same instructions — that needs
the opt-in live harness in ``live_eval.py``.
"""

from smolagents.models import ChatMessage, MessageRole, Model

FENCE = "```"


def code_reply(code: str, thought: str = "I will plan the run and return the preview.") -> str:
    return f"Thought: {thought}\n{FENCE}python\n{code.strip()}\n{FENCE}"


class ScriptedModel(Model):
    """Replay a script of replies; each item is text or a callable(messages)."""

    def __init__(self, replies, model_id: str = "scripted-deterministic"):
        super().__init__(model_id=model_id)
        self._replies = list(replies)
        self.calls = []

    def generate(
        self,
        messages,
        stop_sequences=None,
        response_format=None,
        tools_to_call_from=None,
        **kwargs,
    ):
        recorded = [
            ChatMessage(role=message.role, content=message.content) for message in messages
        ]
        self.calls.append(recorded)
        if not self._replies:
            raise RuntimeError(
                "scripted model has no replies left; the agent asked for more steps "
                f"than the test scripted ({len(self.calls)} calls)"
            )
        reply = self._replies.pop(0)
        content = reply(recorded) if callable(reply) else reply
        return ChatMessage(role=MessageRole.ASSISTANT, content=content)

    @property
    def remaining_replies(self) -> int:
        return len(self._replies)
