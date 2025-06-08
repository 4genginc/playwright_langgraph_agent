import logging
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from state import BrowserState
from browser.playwright_manager import PlaywrightManager

logger = logging.getLogger(__name__)

class WebBrowsingAgent:
    def __init__(self, openai_api_key: str, headless: bool = True):
        # ... initialization

    def _build_graph(self) -> StateGraph:
        # ... build your LangGraph

    # Node handlers:
    async def _initialize_browser(self, state: BrowserState) -> BrowserState:
        # ...

    async def _navigate_to_page(self, state: BrowserState) -> BrowserState:
        # ...
    # (other state node methods...)

    async def execute_task(self, url: str, task: str, task_type: str = "extract", form_data: dict = None) -> dict:
        # ...
