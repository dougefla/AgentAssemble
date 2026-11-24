"""Shared LightPhyx pool lifecycle and prewarm contracts."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

from evo.engine.lightphyx_manager import SHARED_URL_ENV, SharedLightphyxServer


class FakeServer:
    instances = []

    def __init__(self, pool, headless):
        self.pool = pool
        self.headless = headless
        self.url = "ws://shared/ws"
        self.closed = False
        self.instances.append(self)

    def close(self):
        self.closed = True


class FakeClient:
    instances = []

    def __init__(self, url, timeout):
        self.url = url
        self.timeout = timeout
        self.calls = []
        self.closed = False
        self.instances.append(self)

    def call(self, method, **params):
        self.calls.append((method, params))
        if method == "create_envs":
            return {"envs": list(range(params["count"]))}
        if method == "destroy_envs":
            return {"ok": True}
        raise AssertionError(method)

    def close(self):
        self.closed = True


def test_shared_pool_prewarms_capacity_and_restores_env(monkeypatch):
    FakeServer.instances = []
    FakeClient.instances = []
    monkeypatch.delenv(SHARED_URL_ENV, raising=False)
    monkeypatch.setitem(sys.modules, "lightphyx", SimpleNamespace(Client=FakeClient))
    monkeypatch.setitem(
        sys.modules,
        "lightphyx.server",
        SimpleNamespace(ServerProcess=FakeServer),
    )

    with SharedLightphyxServer(50):
        assert os.environ[SHARED_URL_ENV] == "ws://shared/ws"
        assert FakeServer.instances[0].pool == (8, 1, 7)
        assert FakeClient.instances[0].calls == [
            ("create_envs", {"count": 50, "physics": False}),
            ("destroy_envs", {"env": list(range(50))}),
        ]
        assert FakeClient.instances[0].closed is True

    assert SHARED_URL_ENV not in os.environ
    assert FakeServer.instances[0].closed is True


def test_shared_pool_reuses_external_url_without_owning_it(monkeypatch):
    monkeypatch.setenv(SHARED_URL_ENV, "ws://external/ws")
    manager = SharedLightphyxServer(32)
    with manager:
        assert manager.server is None
    assert os.environ[SHARED_URL_ENV] == "ws://external/ws"


def _install_fakes(monkeypatch):
    FakeServer.instances = []
    FakeClient.instances = []
    monkeypatch.delenv(SHARED_URL_ENV, raising=False)
    monkeypatch.setitem(sys.modules, "lightphyx", SimpleNamespace(Client=FakeClient))
    monkeypatch.setitem(
        sys.modules, "lightphyx.server", SimpleNamespace(ServerProcess=FakeServer)
    )


def test_recycle_replaces_server_and_drops_cached_oracle_sim(monkeypatch):
    from evo.engine import lightphyx_manager
    from evo.oracle import render_lightphyx

    _install_fakes(monkeypatch)
    urls = iter(["ws://first/ws", "ws://second/ws"])
    monkeypatch.setattr(
        FakeServer, "__init__",
        lambda self, pool, headless: (
            setattr(self, "pool", pool), setattr(self, "headless", headless),
            setattr(self, "url", next(urls)), setattr(self, "closed", False),
            FakeServer.instances.append(self),
        ) and None,
    )
    dropped = []
    monkeypatch.setattr(render_lightphyx, "_shutdown", lambda: dropped.append(True))

    with SharedLightphyxServer(20) as manager:
        first = manager.server
        assert os.environ[SHARED_URL_ENV] == "ws://first/ws"
        assert lightphyx_manager.recycle_shared_pool("round 2") is True
        assert first.closed is True
        assert manager.server is not first
        assert os.environ[SHARED_URL_ENV] == "ws://second/ws"
        assert dropped == [True]
        # The fresh pool is prewarmed exactly like the first one.
        assert [c.calls[0] for c in FakeClient.instances] == [
            ("create_envs", {"count": 20, "physics": False}),
        ] * 2
        second = manager.server

    assert second.closed is True
    assert lightphyx_manager.recycle_shared_pool() is False  # nothing active


def test_recycle_is_a_noop_for_external_or_disabled_pools(monkeypatch):
    from evo.engine import lightphyx_manager

    monkeypatch.setenv(SHARED_URL_ENV, "ws://external/ws")
    with SharedLightphyxServer(8) as manager:
        assert manager.recycle() is False
        assert lightphyx_manager.recycle_shared_pool() is False
    assert os.environ[SHARED_URL_ENV] == "ws://external/ws"

    monkeypatch.delenv(SHARED_URL_ENV, raising=False)
    with SharedLightphyxServer(8, enabled=False) as manager:
        assert manager.recycle() is False


def test_pool_shape_honours_lightphyx_browsers(monkeypatch):
    from evo.engine.lightphyx_manager import pool_shape

    monkeypatch.delenv("LIGHTPHYX_BROWSERS", raising=False)
    assert pool_shape(65) == (8, 1, 9)
    assert pool_shape(3) == (3, 1, 1)
    monkeypatch.setenv("LIGHTPHYX_BROWSERS", "4")
    assert pool_shape(65) == (4, 1, 17)
    assert pool_shape(2) == (2, 1, 1)      # never more browsers than envs
    for bad in ("0", "-1", "four"):
        monkeypatch.setenv("LIGHTPHYX_BROWSERS", bad)
        with pytest.raises(ValueError, match="LIGHTPHYX_BROWSERS"):
            pool_shape(10)
