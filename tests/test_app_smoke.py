"""Smoke test of the Streamlit pages with offline fakes (no network)."""
import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from streamlit.testing.v1 import AppTest  # noqa: E402

from fakes import PAPERS_SUPPORT, FakeServices  # noqa: E402


@pytest.fixture()
def app(monkeypatch):
    import app_ai
    fake = FakeServices(papers=[{**PAPERS_SUPPORT, "_needs": ["warfarin", "fluconazole"]}])
    monkeypatch.setattr(app_ai, "services", lambda: fake)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
    at.run()
    return at


def test_every_page_renders(app):
    for pg in app.sidebar.radio[0].options:
        app.sidebar.radio[0].set_value(pg).run()
        assert not app.exception, (pg, [e.value for e in app.exception])


def test_mechanism_explorer_end_to_end(app):
    app.sidebar.radio[0].set_value("Mechanism explorer").run()
    app.text_input[0].set_value("warfarin, fluconazole")
    app.button[0].click().run()
    assert not app.exception, [e.value for e in app.exception]
    assert not app.error, [e.value for e in app.error]
    text = " ".join(m.value for m in app.markdown)
    assert "CYP2C9" in text and "SUPPORTED" in text.upper()
    assert any("Evidence for" in e.label for e in app.expander)


def test_structure_page_with_fake_pubchem(app, monkeypatch):
    app.sidebar.radio[0].set_value("Structure & analogues").run()
    app.text_input[0].set_value("warfarin")
    app.text_input[1].set_value("warfarin, fluconazole, pravastatin")
    app.button[0].click().run()
    assert not app.exception, [e.value for e in app.exception]
    assert not app.error, [e.value for e in app.error]
    assert any("Structural look-alikes" in m.value for m in app.markdown)
