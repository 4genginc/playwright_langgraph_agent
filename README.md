# 🦜 Playwright LangGraph Agent

A **pedagogical, modular tutorial and starter template** for building autonomous web-browsing AI agents using Playwright (browser automation), LangGraph (LLM state orchestration), and OpenAI (LLM reasoning).

---

## 📦 1. Directory Layout (Modular Split)

```plaintext
playwright_langgraph_agent/
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
│   └── test_web_browsing_agent.py # Full agent workflow test
└── utils.py                  # Misc. helpers/utilities
```

---

## 📚 2. What Is the Agent? (`agent/web_browsing_agent.py`)

### **Purpose**

* Orchestrates a full web-browsing state machine with LangGraph.
* Delegates browser actions to `PlaywrightManager` (browser/automation logic).
* Injects reasoning and action-selection via `ChatOpenAI` (LLM API).
* Provides an async `.execute_task(url, task, ...)` for end-to-end automation.

### **Pedagogical Highlights**

* **Separation of concerns:** Browser logic vs agent logic vs state vs toolkit.
* **Graph-based orchestration:** Each state node is a function; routing logic is explicit.
* **Extendable:** New agent states or browser actions are easy to add.

---

## 🚀 3. How To Use (Quickstart)

### **Environment Setup**

```bash
# Clone repo & enter dir
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
playwright install  # This downloads browser drivers
```

* Set your OpenAI API key in environment or .env:

  ```bash
  export OPENAI_API_KEY=sk-...
  ```

### **Running the Agent in Code**

```python
from agent.web_browsing_agent import WebBrowsingAgent
import os, asyncio

api_key = os.getenv("OPENAI_API_KEY")
agent = WebBrowsingAgent(api_key, headless=True)
result = asyncio.run(agent.execute_task(
    url="https://news.ycombinator.com",
    task="Extract top 5 headlines and links",
    task_type="extract"
))
print(result)
```

### **Running the CLI (main.py)**

```bash
python main.py
```

* Follow menu prompts for scraping, form filling, custom tasks, etc.

### **Testing**

**Smoke test browser only:**

```bash
pytest tests/test_playwright.py
```

**Test full agent orchestration:**

```bash
pytest tests/test_web_browsing_agent.py
```

---

## 🧠 4. Pedagogical Guide to the Agent

### **Core Components**

* `state.py` - Defines the dataclass for agent state (task, navigation, memory, extracted data, etc.)
* `browser/playwright_manager.py` - Encapsulates all Playwright (browser) actions, keeping browser logic isolated.
* `agent/web_browsing_agent.py` - LLM-powered, LangGraph-based agent orchestration. Handles routing, invokes browser manager, and parses LLM outputs.
* `toolkit/web_toolkit.py` - Batch utilities (e.g., run agent on multiple URLs, export to CSV).

### **Agent Workflow**

1. **Initialize browser** (headless by default)
2. **Navigate** to target URL
3. **Analyze page** with LLM (determines if/what to extract, click, or fill)
4. **Extract data, interact, or search** (based on LLM decision)
5. **Loop/route** until task is complete
6. **Cleanup** browser, return results

### **Testing/Debugging Pedagogy**

* **Direct Playwright test:** Always validate browser automation in isolation (`test_playwright.py`).
* **Agent integration test:** Use async pytest to ensure agent + browser + LLM workflow (`test_web_browsing_agent.py`).
* **Best practice:** Test lower layers before agent layer; stub browser manager for fast tests.

---

## 🏗️ 5. Next Steps & Extending

* Add new agent states/nodes for more actions (file upload, advanced forms, etc.)
* Integrate more LLM providers (Anthropic, Gemini, etc.) via LangChain.
* Batch process with `toolkit/web_toolkit.py` or add database integration.
* Add CI tests and coverage reports.

---

## 📝 6. Example Agent Test (for pytest)

```python
import os, asyncio
from agent.web_browsing_agent import WebBrowsingAgent

def test_web_browsing_agent_extract():
    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key, "OPENAI_API_KEY must be set"

    async def run_agent():
        agent = WebBrowsingAgent(api_key, headless=True)
        result = await agent.execute_task(
            url="https://news.ycombinator.com",
            task="Extract top 3 headlines and their links",
            task_type="extract"
        )
        assert isinstance(result, dict)
        assert "success" in result

    asyncio.run(run_agent())
```

---

## 🤝 7. Attribution & Inspiration

* Inspired by Ed Donner’s LLM engineering course, OpenAI, LangChain, LangGraph, Playwright, and the modern Python ecosystem.

---

**For questions, extensions, or detailed walkthroughs of any file/module, see source or ask the project author!**
