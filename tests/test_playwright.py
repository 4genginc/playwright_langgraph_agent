# playwright_test.py

# tests/test_playwright.py

from playwright.sync_api import sync_playwright
import time

def test_playwright_headless_httpbin():
    start = time.time()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto("https://httpbin.org", timeout=10000)  # 10s timeout
        print("Title:", page.title())
        print("Content (first 500 chars):", page.content()[:500])
        browser.close()
    print("Elapsed:", time.time() - start, "seconds")
