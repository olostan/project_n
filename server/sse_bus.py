"""
Project N: Server-Sent Events (SSE) Event Bus.
Dispatches pipeline stages, memory telemetry, and card events to connected clients.
"""

import asyncio
import json
import time
from collections.abc import AsyncGenerator
from typing import Any


class SSEBus:
    """In-memory Server-Sent Events (SSE) manager with replay buffer."""

    def __init__(self, replay_limit: int = 100) -> None:
        self.subscribers: set[asyncio.Queue[str]] = set()
        self.replay_buffer: list[tuple[int, str, str]] = []  # (event_id, event_type, payload)
        self.replay_limit = replay_limit
        self.counter = 0

    def publish(self, event_type: str, data: dict[str, Any]) -> None:
        """Publishes an event to all connected subscriber queues."""
        self.counter += 1
        payload = json.dumps(data)
        event_str = f"id: {self.counter}\nevent: {event_type}\ndata: {payload}\n\n"

        # Maintain sliding replay buffer
        self.replay_buffer.append((self.counter, event_type, payload))
        if len(self.replay_buffer) > self.replay_limit:
            self.replay_buffer.pop(0)

        # Broadcast to all live queues
        dead_queues = []
        for q in self.subscribers:
            try:
                q.put_nowait(event_str)
            except asyncio.QueueFull:
                dead_queues.append(q)

        for dq in dead_queues:
            self.subscribers.discard(dq)

    async def subscribe(self, last_event_id: int | None = None) -> AsyncGenerator[str, None]:
        """
        Yields events to an SSE client connection.
        Replays missed events if last_event_id is provided.
        Emits periodic keep-alive comments every 15 seconds.
        """
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=100)
        self.subscribers.add(queue)

        try:
            # Replay missed events if requested
            if last_event_id is not None:
                for eid, etype, epayload in self.replay_buffer:
                    if eid > last_event_id:
                        yield f"id: {eid}\nevent: {etype}\ndata: {epayload}\n\n"

            while True:
                try:
                    # Wait up to 15 seconds for an event, then send keep-alive heartbeat
                    msg = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield msg
                except TimeoutError:
                    yield f":keep-alive {int(time.time())}\n\n"
        finally:
            self.subscribers.discard(queue)
