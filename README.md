# 🦜 Playwright LangGraph Agent

A modular, production-grade and pedagogical template for building autonomous, LLM-powered web-browsing agents using Python, Playwright (browser automation), LangGraph (state orchestration), and OpenAI (LLM reasoning).

---

## 📁 Project Directory Layout

```plaintext
playwright_langgraph_agent/
├── .env                     # API keys and environment variables (never commit this!)
├── main.py                   # Entry point; handles CLI/menu logic
├── config.py                 # Env setup, logging, config handling
├── state.py                  # BrowserState dataclass and state definitions
├── agent/
│   └── web_browsing_agent.py # WebBrowsingAgent (LangGraph logic)
├── browser/
│   └── playwright_manager.py # PlaywrightManager (browser automation utils)
├── toolkit/
│   └── web_toolkit.py        # Batch, CSV, etc.
├── examples/
│   └── demo_tasks.py         # Example use cases
├── tests/
│   ├── test_playwright.py    # Playwright browser sanity check
│   ├── test_playwright_manager.py # PlaywrightManager async logic test
│   └── test_web_browsing_agent.py # Full agent workflow test
└── utils.py                  # Misc. helpers/utilities
```

---

## 🚀 Quickstart

1. **Install requirements:**

   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   playwright install  # downloads browser drivers
   ```
2. **Set your OpenAI API key** in `.env` or your shell:

   ```bash
   export OPENAI_API_KEY=sk-...
   ```
3. **Run the CLI agent:**

   ```bash
   python main.py
   ```
4. **Run tests:**

   ```bash
   pytest tests/
   ```

---

## 📚 Module-by-Module Overview

| Module/File                        | Status   | Description                                            |
| ---------------------------------- | -------- | ------------------------------------------------------ |
| `main.py`                          | 🟢 Draft | CLI entry/menu to run agent tasks                      |
| `config.py`                        | 🟢 Done  | Env loading, logging config, project constants         |
| `state.py`                         | 🟢 Done  | Agent state/dataclass for all memory, results, errors  |
| `agent/web_browsing_agent.py`      | 🟢 Done  | Core LLM agent (LangGraph orchestrated, async, tested) |
| `browser/playwright_manager.py`    | 🟢 Done  | Async browser actions—navigate, extract, click, etc.   |
| `toolkit/web_toolkit.py`           | 🟢 Done  | Batch processing, CSV/JSON exports, result aggregation |
| `examples/demo_tasks.py`           | 🟢 Done  | Example workflows for learning/testing                 |
| `tests/test_playwright.py`         | 🟢 Done  | Playwright install/smoke test                          |
| `tests/test_playwright_manager.py` | 🟢 Done  | Modular browser backend test                           |
| `tests/test_web_browsing_agent.py` | 🟢 Done  | Full agent (LLM+browser) pipeline test                 |
| `utils.py`                         | 🟡 TODO  | Misc utilities/helpers                                 |

Legend: 🟢 Done | 🟡 TODO | 🟠 In Progress

---

## 💡 How To Extend

* **Add new agent actions** in `web_browsing_agent.py` and state fields in `state.py`.
* **Batch process** via `toolkit/web_toolkit.py` (write `run_batch`, `export_csv`, etc).
* **Demo workflows**: Place in `examples/demo_tasks.py` for reproducible experiments.
* **Advanced features:** Add support for Anthropic/Gemini LLMs, browser auth, file upload, etc.

---

## 🧑‍💻 Example: CLI Agent Usage (`main.py`)

```
import os
import asyncio
from agent.web_browsing_agent import WebBrowsingAgent
from state import BrowserState

from config import load_env
load_env()

def get_api_key():
    return os.getenv("OPENAI_API_KEY")

async def main():
    api_key = get_api_key()
    if not api_key:
        print("Please set OPENAI_API_KEY")
        return
    agent = WebBrowsingAgent(api_key, headless=True)
    url = input("Enter target URL: ")
    task = input("Enter task description: ")
    task_type = input("Task type (extract/interact/search): ") or "extract"
    result = await agent.execute_task(url, task, task_type)
    print("=== RESULT ===")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())

```

---

## 🧪 Testing

* **Browser layer only:**

  ```bash
  pytest tests/test_playwright.py
  pytest tests/test_playwright_manager.py
  ```
* **Full agent pipeline:**

  ```bash
  pytest tests/test_web_browsing_agent.py
  ```

---

## 📚 Example Workflows (`examples/demo_tasks.py`)

The `examples/demo_tasks.py` module provides ready-to-run demo workflows for:

* News extraction from public sites
* Automated form filling
* Custom page search tasks

**How to use:**

```
from config import load_env
load_env()  # <--- DO THIS FIRST

import os
import asyncio
from agent.web_browsing_agent import WebBrowsingAgent
from toolkit.web_toolkit import export_json

async def demo_news_extraction():
    api_key = os.getenv("OPENAI_API_KEY")
    agent = WebBrowsingAgent(api_key, headless=True)
    result = await agent.execute_task(
        url="https://news.ycombinator.com",
        task="Extract the top 10 news headlines and their links",
        task_type="extract"
    )
    print("Demo News Extraction Result:")
    print(result)
    export_json([result], "demo_news_extraction.json")

async def demo_form_filling():
    """Fill out and submit a sample contact form."""
    api_key = os.getenv("OPENAI_API_KEY")
    agent = WebBrowsingAgent(api_key, headless=True)
    form_data = {"#name": "Alice", "#email": "alice@example.com"}
    result = await agent.execute_task(
        url="https://httpbin.org/forms/post",
        task="Fill out the contact form",
        task_type="interact",
        form_data=form_data
    )
    print("Demo Form Fill Result:")
    print(result)
    export_json([result], "demo_form_fill.json")

if __name__ == "__main__":
    # Choose which demo to run by uncommenting below
    asyncio.run(demo_news_extraction())
    # asyncio.run(demo_form_filling())

```

To run a demo, simply:

```bash
python examples/demo_tasks.py
```

Edit the script to select which demo(s) to run.

---

* See canvas: Pedagogical Guides for `web_browsing_agent.py`, `playwright_manager.py`, and `state.py` for deep-dive documentation.
* [Playwright Python Docs](https://playwright.dev/python/)
* [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
* [LangChain Docs](https://python.langchain.com/)

---

## 🤝 Credits & Inspiration

* Ed Donner (LLM Engineering)
* OpenAI, LangChain, LangGraph, Playwright Python Community
* Modular/agent design patterns from the AI engineering world
