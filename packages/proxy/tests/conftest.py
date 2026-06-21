"""Shared test fixtures. Isolate the store to a temp DB so tests never touch the repo's
vigil.db and the proxy app's lifespan picks up a clean database."""

import os

import pytest

import vigil_proxy.settings as settings_mod


@pytest.fixture(autouse=True, scope="session")
def _isolated_db(tmp_path_factory):
    db = tmp_path_factory.mktemp("vigil") / "test.db"
    os.environ["VIGIL_DB_PATH"] = str(db)
    # Reset the cached settings singleton so the temp DB path is read.
    settings_mod._settings = None
    yield
    settings_mod._settings = None
