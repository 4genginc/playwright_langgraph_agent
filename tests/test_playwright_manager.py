import pytest
from browser.playwright_manager import PlaywrightManager
import asyncio

@pytest.mark.asyncio
async def test_playwright_manager_navigate_extract():
    mgr = PlaywrightManager(headless=True)
    started = await mgr.start()
    assert started, "Browser failed to start"
    info = await mgr.navigate("https://example.com")
    print("Title:", info.get("title"))
    elements = await mgr.extract_elements()
    print("Some elements:", elements)
    await mgr.cleanup()

    # Simple asserts for sanity
    assert "title" in info
    assert info["success"] is True
    assert isinstance(elements, list)
