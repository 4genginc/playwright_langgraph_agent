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
        self.playwright: Playwright | None = None
        self.browser: Browser | None = None
        self.page: Page | None = None

    async def start(self):
        try:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=self.headless,
                args=[
                    '--no-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-blink-features=AutomationControlled',
                    '--disable-extensions'
                ]
            )
            context = await self.browser.new_context(viewport=self.viewport)
            self.page = await context.new_page()
            logger.info("Browser initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize browser: {e}")
            return False

    async def navigate(self, url: str, wait_until: str = 'networkidle') -> dict:
        try:
            response = await self.page.goto(url, wait_until=wait_until, timeout=30000)
            title = await self.page.title()
            content = await self.page.content()
            current_url = self.page.url
            return {
                "success": True,
                "url": current_url,
                "title": title,
                "content": content,
                "status_code": response.status if response else 200
            }
        except Exception as e:
            logger.error(f"Navigation failed for {url}: {e}")
            return {"success": False, "error": str(e), "url": url}

    async def take_screenshot(self, path="screenshot.png"):
        try:
            await self.page.screenshot(path=path, full_page=True)
            logger.info(f"Screenshot saved to {path}")
            return path
        except Exception as e:
            logger.error(f"Screenshot failed: {e}")
            return ""

    async def extract_elements(self, selector: str = None):
        try:
            elements = []
            # Default selectors for common elements
            selectors = [
                'a[href]',
                'button',
                'input',
                'form',
                'h1, h2, h3',
                '[data-testid]',
                '.btn, .button',
            ] if not selector else [selector]
            for sel in selectors:
                try:
                    page_elements = await self.page.query_selector_all(sel)
                    for element in page_elements[:10]:  # Limit to prevent overflow
                        text = await element.text_content() or ""
                        tag_name = await element.evaluate('el => el.tagName.toLowerCase()')
                        attrs = await element.evaluate('''
                            el => {
                                const result = {};
                                if (el.href) result.href = el.href;
                                if (el.id) result.id = el.id;
                                if (el.className) result.class = el.className;
                                if (el.type) result.type = el.type;
                                if (el.name) result.name = el.name;
                                if (el.value) result.value = el.value;
                                return result;
                            }
                        ''')
                        elements.append({
                            "selector": sel,
                            "tag": tag_name,
                            "text": text.strip()[:100],
                            "attributes": attrs
                        })
                except Exception as e:
                    logger.debug(f"Failed to extract {sel}: {e}")
                    continue
            return elements
        except Exception as e:
            logger.error(f"Element extraction failed: {e}")
            return []

    async def click_element(self, selector: str, wait_for_navigation: bool = True):
        try:
            await self.page.wait_for_selector(selector, timeout=10000)
            if wait_for_navigation:
                async with self.page.expect_navigation(timeout=15000):
                    await self.page.click(selector)
            else:
                await self.page.click(selector)
                await self.page.wait_for_timeout(1000)
            return {"success": True, "action": f"clicked {selector}"}
        except Exception as e:
            logger.error(f"Click failed for {selector}: {e}")
            return {"success": False, "error": str(e)}

    async def fill_form(self, form_data: dict):
        results = []
        for selector, value in form_data.items():
            try:
                await self.page.wait_for_selector(selector, timeout=5000)
                await self.page.fill(selector, value)
                results.append({"field": selector, "value": value, "success": True})
            except Exception as e:
                logger.error(f"Failed to fill {selector}: {e}")
                results.append({"field": selector, "success": False, "error": str(e)})
        return {"form_results": results}

    async def search_text(self, text: str):
        try:
            results = await self.page.evaluate(f'''
                () => {{
                    const searchText = "{text}";
                    const walker = document.createTreeWalker(
                        document.body,
                        NodeFilter.SHOW_TEXT
                    );
                    const results = [];
                    let node;
                    while (node = walker.nextNode()) {{
                        if (node.textContent.toLowerCase().includes(searchText.toLowerCase())) {{
                            const element = node.parentElement;
                            results.push({{
                                text: node.textContent.trim(),
                                tagName: element.tagName.toLowerCase(),
                                className: element.className || "",
                                id: element.id || ""
                            }});
                        }}
                    }}
                    return results.slice(0, 10);
                }}
            ''')
            return results
        except Exception as e:
            logger.error(f"Text search failed: {e}")
            return []

    async def cleanup(self):
        try:
            if self.page:
                await self.page.close()
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
            logger.info("Browser cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
```

---

## 🤝 Attributions

* Playwright Python docs: [https://playwright.dev/python/](https://playwright.dev/python/)
* Modular/agent inspiration from LangGraph, OpenAI, and the LLM engineering community.
