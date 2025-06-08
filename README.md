# 🦜 Playwright LangGraph Agent

A modular, production-grade and pedagogical template for building autonomous, LLM-powered web-browsing agents using Python, Playwright (browser automation), LangGraph (state orchestration), and OpenAI (LLM reasoning).

---

## 📁 Project Directory Layout

```plaintext
playwright_langgraph_agent/
├── .env                      # API keys and environment variables (never commit this!)
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
| `toolkit/web_toolkit.py`           | 🟡 TODO  | Batch processing, CSV/JSON exports, result aggregation |
| `examples/demo_tasks.py`           | 🟡 TODO  | Example workflows for learning/testing                 |
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

```python
import os
import asyncio
from agent.web_browsing_agent import WebBrowsingAgent
from state import BrowserState

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

## 📖 Further Learning & Reference

* See canvas: Pedagogical Guides for `web_browsing_agent.py`, `playwright_manager.py`, and `state.py` for deep-dive documentation.
* [Playwright Python Docs](https://playwright.dev/python/)
* [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
* [LangChain Docs](https://python.langchain.com/)

---

## 🤝 Credits & Inspiration

* Ed Donner (LLM Engineering)
* OpenAI, LangChain, LangGraph, Playwright Python Community
* Modular/agent design patterns from the AI engineering world
