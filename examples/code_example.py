import asyncio
import signal
from typing import AsyncGenerator

class AsyncEventPipeline:
    def __init__(self, buffer_size: int = 100):
        self.queue: asyncio.Queue[str] = asyncio.Queue(maxsize=buffer_size)
        self.is_running = True

    async def produce_events(self, source_name: str, count: int) -> None:
        """Produce streaming events into the asynchronous queue."""
        for i in range(count):
            if not self.is_running:
                break
            payload = f"event-{source_name}-{i}"
            await self.queue.put(payload)
            await asyncio.sleep(0.05)

    async def consume_events(self) -> AsyncGenerator[str, None]:
        """Consume events from queue yielding items as they arrive."""
        while self.is_running or not self.queue.empty():
            try:
                item = await asyncio.wait_for(self.queue.get(), timeout=0.2)
                yield item
                self.queue.task_done()
            except asyncio.TimeoutError:
                continue

    def shutdown(self) -> None:
        """Gracefully signal the pipeline to stop consuming."""
        self.is_running = False
