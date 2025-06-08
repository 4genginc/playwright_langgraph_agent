# Playwright LangGraph Agent

## 1. Directory Layout (first-level split)

```
playwright_langgraph_agent/
├── main.py                   # Entry point; handles CLI/menu logic
├── config.py                 # Env setup, logging, config handling
├── state.py                  # BrowserState dataclass and state definitions
├── browser/
│   └── playwright_manager.py # PlaywrightManager (browser automation utils)
├── agent/
│   ├── __init__.py
│   └── web_browsing_agent.py # WebBrowsingAgent (LangGraph logic)
├── toolkit/
│   ├── __init__.py
│   └── web_toolkit.py        # Batch, CSV, etc.
├── examples/
│   ├── __init__.py
│   └── demo_tasks.py         # Example use cases
└── utils.py                  # Misc. helpers/utilities

```

## How to Use

-  ** Import using relative paths** within the package.
    -   E.g., `from browser.playwright_manager import PlaywrightManager`
    -   E.g., `from agent.web_browsing_agent import WebBrowsingAgent`
-   **Start development**:
    Run `main.py` as your CLI entry point.

## Next Steps

-   Add tests (optionally in a `tests`/ directory).
-   Add a `README.md` for developer instructions.

