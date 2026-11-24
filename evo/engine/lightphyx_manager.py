"""One shared LightPhyx render pool for an evo process.

Launching one Chromium stack inside every sim-worker is reliable at small
parallelism but creates a browser-startup thundering herd at 32+ workers.  The
engine already supports many isolated envs per page, so the training parent
owns one pre-warmed server and workers connect to it through an inherited URL.
"""
from __future__ import annotations

import math
import os
from types import TracebackType


SHARED_URL_ENV = "EVO_LIGHTPHYX_URL"
MAX_BROWSERS = 8
# Browser count override (same variable the lightphyx server reads). Each
# browser is one GPU process of ~0.75 GB; fewer browsers save memory, more
# render in parallel. Unset: min(MAX_BROWSERS, capacity).
BROWSERS_ENV = "LIGHTPHYX_BROWSERS"


def pool_shape(capacity: int) -> tuple[int, int, int]:
    """(browsers, pages per browser, envs per page) holding `capacity` envs."""
    raw = os.environ.get(BROWSERS_ENV, "").strip()
    if raw:
        try:
            browsers = int(raw)
        except ValueError:
            browsers = 0
        if browsers < 1:
            raise ValueError(f"{BROWSERS_ENV} must be a positive integer, got {raw!r}")
        browsers = min(browsers, capacity)
    else:
        browsers = min(MAX_BROWSERS, capacity)
    return browsers, 1, math.ceil(capacity / browsers)

# The pool owned by the running loop (None outside `with SharedLightphyxServer`).
_ACTIVE: "SharedLightphyxServer | None" = None


def recycle_shared_pool(reason: str = "") -> bool:
    """Restart the loop-owned LightPhyx pool between rounds.

    Headless Chromium's GPU process accumulates memory over a long run (mesh
    templates, render targets); on a 24 GB Mac a 1.5 h run pushed eight GPU
    processes to ~16 GB and the machine into swap. Restarting the browsers at
    a round boundary — when no executor, sim worker or oracle render is in
    flight — bounds that growth to one round. Returns False when there is no
    loop-owned pool to recycle (disabled, or reusing an external server).
    """
    if _ACTIVE is None:
        return False
    return _ACTIVE.recycle(reason)


class SharedLightphyxServer:
    """Context manager for a pre-warmed B×1×E LightPhyx pool."""

    def __init__(self, capacity: int, *, enabled: bool = True):
        self.capacity = max(1, int(capacity))
        self.enabled = enabled
        self.server = None
        self._previous_url: str | None = None
        self._reusing_external = False

    def start(self) -> None:
        if not self.enabled:
            return
        existing = os.environ.get(SHARED_URL_ENV)
        if existing:
            self._reusing_external = True
            print(f"LightPhyx pool: reusing {existing}", flush=True)
            return

        self._previous_url = existing
        self._launch()
        global _ACTIVE
        _ACTIVE = self

    def _launch(self) -> None:
        from lightphyx import Client
        from lightphyx.server import ServerProcess

        browsers, _, envs_per_page = pool_shape(self.capacity)
        try:
            self.server = ServerProcess(
                pool=(browsers, 1, envs_per_page), headless=True
            )
            # Force all browser/pages to launch before any executor starts.
            # Destroying these temporary envs preserves the warm pages and
            # returns their capacity to the pool for worker-owned envs.
            client = Client(
                self.server.url,
                timeout=max(120.0, float(self.capacity) * 5.0),
            )
            try:
                result = client.call(
                    "create_envs", count=self.capacity, physics=False,
                )
                env_ids = list(result.get("envs", []))
                if len(env_ids) != self.capacity:
                    raise RuntimeError(
                        "LightPhyx prewarm created "
                        f"{len(env_ids)}/{self.capacity} envs"
                    )
                client.call("destroy_envs", env=env_ids)
            finally:
                client.close()
            os.environ[SHARED_URL_ENV] = self.server.url
            print(
                "LightPhyx pool: "
                f"capacity={self.capacity}, browsers={browsers}, "
                f"envs_per_page={envs_per_page}, url={self.server.url}",
                flush=True,
            )
        except BaseException:
            if self.server is not None:
                self.server.close()
                self.server = None
            raise

    def recycle(self, reason: str = "") -> bool:
        """Replace the running server (and its browsers) with a fresh one.

        Only safe when nothing is rendering: callers use it at round
        boundaries. The new server gets a new port, so the shared URL env var
        is updated for children spawned afterwards, and this process's cached
        oracle connection is dropped so it reconnects on next use.
        """
        if self._reusing_external or not self.enabled or self.server is None:
            return False
        # The loop process's oracle renderer caches one Sim on the old URL.
        from evo.oracle.render_lightphyx import _shutdown as drop_cached_sim

        drop_cached_sim()
        old = self.server
        self.server = None
        old.close()
        print(
            f"LightPhyx pool: recycling browsers{f' ({reason})' if reason else ''}",
            flush=True,
        )
        self._launch()
        return True

    def stop(self) -> None:
        global _ACTIVE
        if _ACTIVE is self:
            _ACTIVE = None
        if self._reusing_external or not self.enabled:
            return
        if self._previous_url is None:
            os.environ.pop(SHARED_URL_ENV, None)
        else:
            os.environ[SHARED_URL_ENV] = self._previous_url
        if self.server is not None:
            self.server.close()
            self.server = None

    def __enter__(self) -> "SharedLightphyxServer":
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()
