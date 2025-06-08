import os
import asyncio
from agent.web_browsing_agent import WebBrowsingAgent
from state import BrowserState
from config import load_env, get_api_key, setup_logging

load_env()
setup_logging("INFO")

api_key = get_api_key("OPENAI_API_KEY")

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
