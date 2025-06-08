# 🧑‍💻 Pedagogical Guide: `browser/playwright_manager.py`

A **pedagogical, modular tutorial and starter template** for the `PlaywrightManager` class in your Playwright LangGraph Agent project. This guide walks through the architecture, methods, and best practices of your browser automation backend.

---

## 📂 Where Does This Fit?

```plaintext
playwright_langgraph_agent/
└── browser/
    └── playwright_manager.py   # This file
```

* **Role:** Encapsulates browser actions for your agent. All web automation is delegated here.

---

## 🎯 Purpose and Design Philosophy

* **Single Responsibility:** All Playwright logic is kept in one place.
* **Async API:** Methods are async, suitable for concurrent, agent-based workloads.
* **Agent-Friendly:** All actions return dictionaries/lists (never throw unhandled exceptions), so they are LLM- and agent-safe.
* **Easy to Extend:** Add new browser actions without changing agent logic.

---

## 🚦 Typical Usage Pattern

```python
from browser.playwright_manager import PlaywrightManager
import asyncio

async def demo():
    mgr = PlaywrightManager(headless=True)
    await mgr.start()
    result = await mgr.navigate("https://example.com")
    elements = await mgr.extract_elements()
    await mgr.cleanup()
    print(result)
    print(elements)

asyncio.run(demo())
```

---

## 🧩 Main Methods – Pedagogical Code Review

### 1. **Initialization**

```python
    def __init__(self, headless=True, viewport_width=1280, viewport_height=720):
        # Store configuration for browser launch
```

* Sets up headless mode (show/hide browser UI), viewport size, but does not launch the browser yet.

### 2. **Start (Browser Session)**

```python
    async def start(self):
        try:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(...)
            ...
            return True
        except Exception as e:
            ...
            return False
```

* Launches a Chromium browser, creates a new page. Logs errors, never throws.

### 3. **Navigate to URL**

```python
    async def navigate(self, url: str, wait_until: str = 'networkidle') -> dict:
        ...
        return {"success": True, ...}  # or {"success": False, ...}
```

* Loads a web page. Returns title, HTML, URL, HTTP status, or a clear error if it fails.

### 4. **Take Screenshot**

```python
    async def take_screenshot(self, path="screenshot.png"):
        ...
        return path
```

* Captures a full-page PNG. Returns path on success, "" on error.

### 5. **Extract Elements**

```python
    async def extract_elements(self, selector: str = None):
        ...
        return elements
```

* Finds links, buttons, inputs, headers, etc. Returns a list of dicts with text, tag, and attributes.
* Custom selector supported.

### 6. **Click Element**

```python
    async def click_element(self, selector: str, wait_for_navigation: bool = True):
        ...
        return {"success": True/False, ...}
```

* Clicks an element, waits for navigation if needed. Safe return value.

### 7. **Fill Form Fields**

```python
    async def fill_form(self, form_data: dict):
        ...
        return {"form_results": ...}
```

* Fills multiple fields by selector. Each result records success/error for that field.

### 8. **Search for Text**

```python
    async def search_text(self, text: str):
        ...
        return results  # List of matches
```

* Searches page for a substring. Returns structured list.

### 9. **Cleanup/Shutdown**

```python
    async def cleanup(self):
        ...
```

* Closes the page, browser, and Playwright context. Ensures no orphaned processes.

---

## 🌟 Pedagogical Tips

* \*\*Always call \*\*\`\` after your work to avoid zombie processes.
* **Use explicit error handling:** All methods return data or error, never crash the agent.
* **Limit extraction/clicks** to avoid overwhelming the browser (sample 10 elements at a time).
* **Logging:** Use `logger` for every major step for debuggability.
* **Async is critical** for scaling up or batching web tasks.

---

## 🔬 Extension Ideas

* Add file upload, drag-drop, mouse actions, custom wait conditions, screenshots of sub-elements.
* Add browser "profiles" or authenticated sessions.
* Integrate human-like delays or anti-bot features for production scraping.

---

## 📚 Example: Full Minimal File Template

```python
import logging
from playwright.async_api import async_playwright, Browser, Page, Playwright

logger = logging.getLogger(__name__)

class PlaywrightManager:
    def __init__(self, headless=True, viewport_width=1280, viewport_height=720):
        self.headless = headless
        self.viewport = {"width": viewport_width, "height": viewport_height}
        self.playwright = None
        self.browser = None
        self.page = None

    async def start(self):
        ... # see above

    async def navigate(self, url: str, wait_until: str = 'networkidle') -> dict:
        ...

    async def take_screenshot(self, path="screenshot.png"):
        ...

    async def extract_elements(self, selector: str = None):
        ...

    async def click_element(self, selector: str, wait_for_navigation: bool = True):
        ...

    async def fill_form(self, form_data: dict):
        ...

    async def search_text(self, text: str):
        ...

    async def cleanup(self):
        ...
```

---

## 🤝 Attributions

* Playwright Python docs: [https://playwright.dev/python/](https://playwright.dev/python/)
* Modular/agent inspiration from LangGraph, OpenAI, and the LLM engineering community.
