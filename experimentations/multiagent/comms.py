"""Communication primitives for experimental multi-agent SimWorld tasks."""

from __future__ import annotations

from collections import defaultdict, deque
from math import hypot
from typing import Any

from pydantic import BaseModel, Field


ALL_RECIPIENTS = "all"


class Message(BaseModel):
    """One addressed communication intent emitted by an agent."""

    sender: str
    recipients: list[str] | None = None
    content: str
    step: int
    delivered_to: list[str] = Field(default_factory=list)

    def is_broadcast(self) -> bool:
        """Return whether this message should be broadcast."""

        if not self.recipients:
            return True
        return any(recipient == ALL_RECIPIENTS for recipient in self.recipients)

    def compact(self) -> dict[str, Any]:
        """Return a compact JSON-serializable representation."""

        return {
            "sender": self.sender,
            "recipients": self.recipients or [ALL_RECIPIENTS],
            "content": self.content,
            "step": self.step,
            "delivered_to": self.delivered_to,
        }


class CommsError(BaseModel):
    """Non-fatal communication validation or delivery issue."""

    step: int
    sender: str
    recipients: list[str] | None
    error: str

    def compact(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return self.model_dump() if hasattr(self, "model_dump") else self.dict()


class CommsRouter:
    """Base class for turning communication intents into per-agent inboxes."""

    def __init__(self, *, max_content_chars: int = 240):
        self.max_content_chars = max_content_chars

    def resolve_recipients(
        self,
        message: Message,
        agent_ids: list[str],
        positions: dict[str, tuple[float, float]] | None = None,
    ) -> tuple[list[str], list[CommsError]]:
        """Return intended recipients after validation."""

        errors: list[CommsError] = []
        if message.sender not in agent_ids:
            errors.append(
                CommsError(
                    step=message.step,
                    sender=message.sender,
                    recipients=message.recipients,
                    error="unknown sender",
                )
            )
            return [], errors

        if message.is_broadcast():
            return [agent_id for agent_id in agent_ids if agent_id != message.sender], errors

        recipients: list[str] = []
        for recipient in message.recipients or []:
            if recipient == message.sender:
                errors.append(
                    CommsError(
                        step=message.step,
                        sender=message.sender,
                        recipients=message.recipients,
                        error=f"self recipient dropped: {recipient}",
                    )
                )
                continue
            if recipient not in agent_ids:
                errors.append(
                    CommsError(
                        step=message.step,
                        sender=message.sender,
                        recipients=message.recipients,
                        error=f"unknown recipient dropped: {recipient}",
                    )
                )
                continue
            if recipient not in recipients:
                recipients.append(recipient)
        return recipients, errors

    def _clean_message(self, message: Message) -> Message | None:
        """Normalize message text and drop empty messages."""

        content = (message.content or "").strip()
        if not content:
            return None
        if len(content) > self.max_content_chars:
            content = content[: self.max_content_chars].rstrip()
        return Message(
            sender=message.sender,
            recipients=message.recipients,
            content=content,
            step=message.step,
        )

    def deliver(
        self,
        messages: list[Message],
        agent_ids: list[str],
        positions: dict[str, tuple[float, float]] | None = None,
    ) -> tuple[dict[str, list[Message]], list[Message], list[CommsError]]:
        """Deliver messages to each agent inbox."""

        inboxes: dict[str, list[Message]] = {agent_id: [] for agent_id in agent_ids}
        transcript: list[Message] = []
        errors: list[CommsError] = []
        for raw_message in messages:
            message = self._clean_message(raw_message)
            if message is None:
                continue
            recipients, message_errors = self.resolve_recipients(message, agent_ids, positions)
            errors.extend(message_errors)
            message.delivered_to = recipients
            transcript.append(message)
            for recipient in recipients:
                inboxes[recipient].append(message)
        return inboxes, transcript, errors


class BroadcastRouter(CommsRouter):
    """Route every valid message to every other agent."""

    def resolve_recipients(
        self,
        message: Message,
        agent_ids: list[str],
        positions: dict[str, tuple[float, float]] | None = None,
    ) -> tuple[list[str], list[CommsError]]:
        errors: list[CommsError] = []
        if message.sender not in agent_ids:
            errors.append(
                CommsError(
                    step=message.step,
                    sender=message.sender,
                    recipients=message.recipients,
                    error="unknown sender",
                )
            )
            return [], errors
        return [agent_id for agent_id in agent_ids if agent_id != message.sender], errors


class DirectedRouter(CommsRouter):
    """Route messages to their validated recipients, with broadcast fallback."""


class ProximityRouter(DirectedRouter):
    """Route directed messages only to recipients within a communication radius."""

    def __init__(self, *, radius: float, max_content_chars: int = 240):
        super().__init__(max_content_chars=max_content_chars)
        self.radius = radius

    def resolve_recipients(
        self,
        message: Message,
        agent_ids: list[str],
        positions: dict[str, tuple[float, float]] | None = None,
    ) -> tuple[list[str], list[CommsError]]:
        recipients, errors = super().resolve_recipients(message, agent_ids, positions)
        if positions is None or message.sender not in positions:
            return recipients, errors

        sender_position = positions[message.sender]
        reachable: list[str] = []
        for recipient in recipients:
            recipient_position = positions.get(recipient)
            if recipient_position is None:
                continue
            distance = hypot(sender_position[0] - recipient_position[0], sender_position[1] - recipient_position[1])
            if distance <= self.radius:
                reachable.append(recipient)
            else:
                errors.append(
                    CommsError(
                        step=message.step,
                        sender=message.sender,
                        recipients=message.recipients,
                        error=f"recipient out of range: {recipient}",
                    )
                )
        return reachable, errors


class MessageBus:
    """Stateful per-agent inbox window plus full communication transcript."""

    def __init__(self, agent_ids: list[str], router: CommsRouter | None = None, max_history: int = 8):
        self.agent_ids = list(agent_ids)
        self.router = router or BroadcastRouter()
        self.max_history = max_history
        self.inboxes: dict[str, deque[Message]] = {agent_id: deque(maxlen=max_history) for agent_id in agent_ids}
        self.transcript: list[Message] = []
        self.errors: list[CommsError] = []

    def reset(self, agent_ids: list[str] | None = None) -> None:
        """Clear delivery state for a new episode."""

        if agent_ids is not None:
            self.agent_ids = list(agent_ids)
        self.inboxes = {agent_id: deque(maxlen=self.max_history) for agent_id in self.agent_ids}
        self.transcript = []
        self.errors = []

    def deliver(
        self,
        messages: list[Message],
        positions: dict[str, tuple[float, float]] | None = None,
    ) -> dict[str, list[Message]]:
        """Deliver one step's messages and update inbox windows."""

        delivered, transcript, errors = self.router.deliver(messages, self.agent_ids, positions)
        self.transcript.extend(transcript)
        self.errors.extend(errors)
        for agent_id, inbox_messages in delivered.items():
            self.inboxes[agent_id].extend(inbox_messages)
        return {agent_id: list(inbox) for agent_id, inbox in self.inboxes.items()}

    def snapshot(self) -> dict[str, Any]:
        """Return the full communication state for logs."""

        return {
            "transcript": [message.compact() for message in self.transcript],
            "errors": [error.compact() for error in self.errors],
            "inboxes": {
                agent_id: [message.compact() for message in messages]
                for agent_id, messages in self.inboxes.items()
            },
        }


def messages_from_turns(
    turns: dict[str, Any],
    *,
    step: int,
    default_recipients: list[str] | None = None,
) -> list[Message]:
    """Extract optional messages from parsed agent turns."""

    messages: list[Message] = []
    for sender, turn in turns.items():
        content = getattr(turn, "message", None)
        if not content:
            continue
        recipients = getattr(turn, "recipients", None) or default_recipients
        messages.append(Message(sender=sender, recipients=recipients, content=content, step=step))
    return messages


def group_messages_by_recipient(messages: list[Message]) -> dict[str, list[Message]]:
    """Group messages by delivered recipient for convenience."""

    grouped: dict[str, list[Message]] = defaultdict(list)
    for message in messages:
        for recipient in message.delivered_to:
            grouped[recipient].append(message)
    return dict(grouped)
