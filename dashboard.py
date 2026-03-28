"""Real-time training dashboard: collector + aiohttp server."""

import asyncio
import json
import os
import threading
import time
from collections import defaultdict, deque

from aiohttp import web


class DashboardCollector:
    """Thread-safe event buffer for training instrumentation."""

    def __init__(self, maxlen=50_000, total_timesteps=0):
        self._events = deque(maxlen=maxlen)
        self._cursor = 0  # monotonic event counter
        self._lock = threading.Lock()
        self.stats = {
            "games": 0, "victories": 0, "defeats": 0,
            "total_reward": 0.0, "total_rounds": 0,
            "timesteps": 0, "total_timesteps": total_timesteps,
        }

    def emit(self, event_type, env_id, **data):
        event = {
            "ts": time.time(), "type": event_type,
            "env_id": env_id, **data,
        }
        with self._lock:
            self._events.append((self._cursor, event))
            self._cursor += 1

            if event_type == "end_turn":
                self.stats["total_rounds"] += 1
                self.stats["total_reward"] += data.get("reward", 0)
                if data.get("terminated"):
                    self.stats["games"] += 1
                    gr = data.get("game_result")
                    if gr == "victory":
                        self.stats["victories"] += 1
                    elif gr == "defeat":
                        self.stats["defeats"] += 1

    def drain_since(self, since_cursor):
        """Return (new_cursor, [events]) for events after since_cursor."""
        with self._lock:
            result = []
            for cursor, event in self._events:
                if cursor > since_cursor:
                    result.append(event)
            return self._cursor, result

    def inc_steps(self, n=1):
        with self._lock:
            self.stats["timesteps"] += n

    def get_stats(self):
        with self._lock:
            return dict(self.stats)


# ── aiohttp server ──

DASHBOARD_HTML = os.path.join(os.path.dirname(__file__), "dashboard.html")


async def index_handler(request):
    return web.FileResponse(DASHBOARD_HTML)


async def pool_handler(request):
    board_pool = request.app["board_pool"]
    if board_pool is None:
        return web.json_response({})
    snapshot = board_pool.snapshot()
    # Convert tuple keys to strings for JSON
    data = {f"{r},{w},{l}": count for (r, w, l), count in snapshot.items()}
    return web.json_response(data)


async def stats_handler(request):
    collector = request.app["collector"]
    return web.json_response(collector.get_stats())


async def ws_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    request.app["ws_clients"].add(ws)
    try:
        async for msg in ws:
            pass  # client doesn't send anything meaningful
    finally:
        request.app["ws_clients"].discard(ws)
    return ws


async def push_loop(app):
    """Background task: push new events + pool snapshots to WebSocket clients."""
    collector = app["collector"]
    board_pool = app["board_pool"]
    cursor = 0
    pool_tick = 0

    while True:
        await asyncio.sleep(0.5)
        if not app["ws_clients"]:
            continue

        new_cursor, events = collector.drain_since(cursor)
        cursor = new_cursor

        payload = {}
        if events:
            payload["events"] = events
        payload["stats"] = collector.get_stats()

        pool_tick += 1
        if pool_tick >= 4 and board_pool is not None:  # every 2 seconds
            pool_tick = 0
            snapshot = board_pool.snapshot()
            payload["pool"] = {f"{r},{w},{l}": count for (r, w, l), count in snapshot.items()}

        if payload:
            msg = json.dumps(payload, default=str)
            dead = set()
            for ws in app["ws_clients"]:
                try:
                    await ws.send_str(msg)
                except Exception:
                    dead.add(ws)
            app["ws_clients"] -= dead


async def on_startup(app):
    app["push_task"] = asyncio.ensure_future(push_loop(app))


async def on_cleanup(app):
    app["push_task"].cancel()


def start_dashboard(collector, board_pool, port=8050):
    """Launch the dashboard server in a daemon thread."""
    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        app = web.Application()
        app["collector"] = collector
        app["board_pool"] = board_pool
        app["ws_clients"] = set()

        app.router.add_get("/", index_handler)
        app.router.add_get("/api/pool", pool_handler)
        app.router.add_get("/api/stats", stats_handler)
        app.router.add_get("/ws", ws_handler)

        app.on_startup.append(on_startup)
        app.on_cleanup.append(on_cleanup)

        runner = web.AppRunner(app)
        loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, "127.0.0.1", port, reuse_port=True)
        try:
            loop.run_until_complete(site.start())
        except OSError:
            # Port in use — try next port
            site = web.TCPSite(runner, "127.0.0.1", port + 1, reuse_port=True)
            loop.run_until_complete(site.start())
        loop.run_forever()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    print(f"  Dashboard: http://localhost:{port}")
