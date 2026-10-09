"""Optional browser check: uv run --with playwright pytest tests/test_chess_games_browser.py.

Requires `uv run --with playwright playwright install chromium` once.
Uses Trackio's iframe sandbox permissions, without starting a training process.
"""

from html import escape
from pathlib import Path

import pytest

playwright = pytest.importorskip("playwright.sync_api")

from nanoalphazero.diagnostics.chess_games import games_html


def test_viewer_navigation_and_downloads_in_sandbox():
    pgn = ('[Event "</script><script>alert(1)</script>"]\n\n'
           '1. e4 a6 2. e5 d5 3. exd6 *\n\n'
           '[SetUp "1"]\n[FEN "7k/P7/8/8/8/8/8/7K w - - 0 1"]\n\n1. a8=Q+ *\n\n')
    with playwright.sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(accept_downloads=True)
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("dialog", lambda d: (errors.append(d.message), d.dismiss()))
        page.set_content('<iframe style="width:600px;height:420px" sandbox="allow-scripts allow-downloads" '
                         f'srcdoc="{escape(games_html(pgn), quote=True)}"></iframe>')
        frame = page.frame_locator("iframe")
        frame.locator("#next").wait_for()
        assert frame.locator("#game option").count() == 2
        assert frame.locator("body").evaluate("() => window.scrollY") == 0
        frame.locator("#next").click()
        assert frame.locator('[data-square="e4"]').get_attribute("data-piece") == "P"
        frame.locator("#last").click()
        assert frame.locator('[data-square="d6"]').get_attribute("data-piece") == "P"
        assert frame.locator('[data-square="d5"]').get_attribute("data-piece") == ""
        frame.locator("#flip").click()
        assert frame.locator(".square").first.get_attribute("data-square") == "h1"
        frame.locator("#game").select_option("1")
        frame.locator("#last").click()
        assert frame.locator('[data-square="a8"]').get_attribute("data-piece") == "Q"
        frame.locator('#moves button[data-ply="0"]').click()
        assert frame.locator('[data-square="a7"]').get_attribute("data-piece") == "P"
        with page.expect_download() as download:
            frame.locator("#download-all").click()
        assert Path(download.value.path()).read_text() == pgn
        frame.get_by_text("Selected game PGN", exact=True).click()
        with page.expect_download() as download:
            frame.locator("#download-game").click()
        assert Path(download.value.path()).read_text() == frame.locator("#selected-pgn").input_value()
        assert not errors
        browser.close()
