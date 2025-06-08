# 📄 Example Demo Tasks for Playwright LangGraph Agent

This file contains sample code and instructions for running demo workflows with your LLM-powered autonomous web-browsing agent. These demos are designed to be pedagogical, modular, and easy to extend!

---

## 🧑‍💻 Example: `examples/demo_tasks.py`

```python
from config import load_env
load_env()  # <--- Ensures .env is loaded

import os
import asyncio
from agent.web_browsing_agent import WebBrowsingAgent
from toolkit.web_toolkit import export_json

async def demo_news_extraction():
    """Extract top headlines from Hacker News."""
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

---

## 🚀 Running Demos

Run either or both demo functions by uncommenting the appropriate line in the `__main__` block.

```bash
python examples/demo_tasks.py
```

* Results are printed to console and saved as JSON files for reproducibility.
* Edit the script to add your own custom demo workflows!

---

## 🔗 References & Docs

* See the main project README for full setup instructions.
* [Playwright Python Docs](https://playwright.dev/python/)
* [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
* [LangChain Docs](https://python.langchain.com/)

---

**Happy automating!** 🚀
q