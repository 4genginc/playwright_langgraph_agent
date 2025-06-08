import logging
import json
import asyncio
from datetime import datetime
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from state import BrowserState
from browser.playwright_manager import PlaywrightManager

logger = logging.getLogger(__name__)

class WebBrowsingAgent:
    def __init__(self, openai_api_key: str, headless: bool = True):
        self.llm = ChatOpenAI(
            model="gpt-4",
            api_key=openai_api_key,
            temperature=0.1,
            max_tokens=1500
        )
        self.browser = PlaywrightManager(headless=headless)
        self.graph = self._build_graph()

    def _ensure_state(self, candidate, fallback):
        # Defensive: Make sure handlers always return BrowserState
        if isinstance(candidate, BrowserState):
            return candidate
        if isinstance(candidate, dict):
            # Attempt to convert from dict to BrowserState
            try:
                return BrowserState(**candidate)
            except Exception as e:
                logger.error(f"Could not coerce dict to BrowserState: {e}")
        logger.warning(f"Handler returned unexpected type: {type(candidate)}. Returning previous state.")
        return fallback

    def _build_graph(self) -> StateGraph:
        def route_after_initialization(state: BrowserState) -> str:
            if state.error_message:
                return "handle_error"
            return "navigate"

        def route_after_navigation(state: BrowserState) -> str:
            if state.error_message:
                return "handle_error"
            return "analyze_page"

        def route_after_analysis(state: BrowserState) -> str:
            if state.error_message:
                return "handle_error"
            elif state.task_completed:
                return "complete_task"
            elif state.task_type == "extract":
                return "extract_data"
            elif state.task_type == "interact":
                return "interact_with_page"
            elif state.task_type == "search":
                return "search_content"
            else:
                return "extract_data"

        def route_after_interaction(state: BrowserState) -> str:
            if state.error_message and state.retry_count >= state.max_retries:
                return "handle_error"
            elif state.task_completed:
                return "complete_task"
            else:
                return "analyze_page"

        graph = StateGraph(BrowserState)
        # Register async node handlers directly (no lambda!)
        graph.add_node("initialize_browser", self._initialize_browser)
        graph.add_node("navigate", self._navigate_to_page)
        graph.add_node("analyze_page", self._analyze_page)
        graph.add_node("extract_data", self._extract_data)
        graph.add_node("interact_with_page", self._interact_with_page)
        graph.add_node("search_content", self._search_content)
        graph.add_node("complete_task", self._complete_task)
        graph.add_node("handle_error", self._handle_error)

        graph.set_entry_point("initialize_browser")
        graph.add_conditional_edges("initialize_browser", route_after_initialization)
        graph.add_conditional_edges("navigate", route_after_navigation)
        graph.add_conditional_edges("analyze_page", route_after_analysis)
        graph.add_conditional_edges("extract_data", route_after_interaction)
        graph.add_conditional_edges("interact_with_page", route_after_interaction)
        graph.add_conditional_edges("search_content", route_after_interaction)
        graph.add_edge("complete_task", END)
        graph.add_edge("handle_error", END)
        return graph.compile()

    # Patch: Call _ensure_state at the end of every handler to ensure state type.
    async def _initialize_browser(self, state: BrowserState) -> BrowserState:
        logger.info("Initializing browser...")
        success = await self.browser.start()
        if success:
            state.current_step = "browser_ready"
            state.navigation_history.append("Browser initialized successfully")
        else:
            state.error_message = "Failed to initialize browser"
        return self._ensure_state(state, state)

    async def _navigate_to_page(self, state: BrowserState) -> BrowserState:
        logger.info(f"Navigating to: {state.target_url}")
        result = await self.browser.navigate(state.target_url)
        if result["success"]:
            state.current_url = result["url"]
            state.page_title = result["title"]
            state.page_content = result["content"][:2000]
            state.current_step = "page_loaded"
            state.navigation_history.append(f"Successfully navigated to {result['url']}")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = f"screenshot_{timestamp}.png"
            state.screenshot_path = await self.browser.take_screenshot(screenshot_path)
        else:
            state.error_message = result["error"]
            state.retry_count += 1
        return self._ensure_state(state, state)

    async def _analyze_page(self, state: BrowserState) -> BrowserState:
        logger.info("Analyzing page content... [START]")
        try:
            state.page_elements = await self.browser.extract_elements()
            logger.debug(f"Extracted page_elements: {state.page_elements!r}")

            elements_summary = [
                f"- {elem.get('tag', '?')}: {str(elem.get('text', ''))[:50]}..."
                for elem in (state.page_elements or [])[:15]
            ]
            logger.debug(f"elements_summary: {elements_summary!r}")

            analysis_prompt = f"""
            TASK: {state.task_description}
            TASK TYPE: {state.task_type}
            CURRENT PAGE:
            Title: {state.page_title}
            URL: {state.current_url}
            AVAILABLE ELEMENTS:
            {chr(10).join(elements_summary)}
            PAGE CONTENT PREVIEW:
            {state.page_content[:800]}
            Based on the task and available page elements, determine:
            1. Is the task complete? (yes/no)
            2. What should be the next action?
            3. Any specific elements to interact with?
            Respond in JSON format:
            {{
                "task_completed": true/false,
                "next_action": "extract/interact/search/complete",
                "reasoning": "...",
                "target_elements": ["selector1", "selector2"],
                "confidence": 0.0-1.0
            }}
            """
            logger.debug(f"analysis_prompt: {analysis_prompt}")

            messages = [
                SystemMessage(content="You are a web browsing assistant. Analyze pages and make decisions based on the given task."),
                HumanMessage(content=analysis_prompt)
            ]
            logger.info("Invoking LLM... (self.llm.ainvoke)")
            response = await self.llm.ainvoke(messages)
            logger.debug(f"LLM response: {repr(response)}")
            logger.debug(f"LLM response.content: {getattr(response, 'content', None)}")

            if response is None:
                logger.error("LLM returned None (response is None)")
                state.error_message = "LLM returned None"
                return self._ensure_state(state, state)
            if getattr(response, "content", None) is None:
                logger.error("LLM returned object with .content == None")
                state.error_message = "LLM returned .content == None"
                return self._ensure_state(state, state)

            try:
                decision = json.loads(response.content)
                logger.debug(f"Parsed LLM JSON: {decision}")
                state.task_completed = decision.get("task_completed", False)
                state.task_type = decision.get("next_action", "extract")
                state.click_targets = decision.get("target_elements", [])
                reasoning = decision.get("reasoning", "No reasoning provided")
                state.navigation_history.append(f"Analysis: {reasoning}")
                logger.info(f"Analysis complete. Next action: {state.task_type}")
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}. Raw LLM response: {response.content!r}")
                content = response.content.lower()
                if "complete" in content or "finished" in content:
                    state.task_completed = True
                elif "click" in content or "button" in content:
                    state.task_type = "interact"
                else:
                    state.task_type = "extract"
                state.navigation_history.append(f"Analysis (fallback): {response.content[:100]}")
        except Exception as e:
            logger.error(f"Page analysis failed: {e}", exc_info=True)
            state.error_message = f"Analysis failed: {str(e)}"
        logger.info("Analyzing page content... [END]")
        return self._ensure_state(state, state)

    async def _extract_data(self, state: BrowserState) -> BrowserState:
        logger.info("Extracting data from page...")
        try:
            extracted = {
                "title": state.page_title,
                "url": state.current_url,
                "timestamp": datetime.now().isoformat(),
                "elements": []
            }
            # Defensive: Ensure state.page_elements is a list
            page_elements = state.page_elements if isinstance(state.page_elements, list) else []
            for element in page_elements:
                if element.get("text") and len(element["text"].strip()) > 5:
                    extracted["elements"].append({
                        "type": element.get("tag", ""),
                        "text": element.get("text", ""),
                        "attributes": element.get("attributes", {})
                    })
            tables = await self.browser.extract_elements("table")
            if tables and isinstance(tables, list):
                extracted["tables"] = [{"text": t.get("text", "")} for t in tables[:3]]
            lists = await self.browser.extract_elements("ul, ol")
            if lists and isinstance(lists, list):
                extracted["lists"] = [{"text": l.get("text", "")} for l in lists[:3]]
            state.extracted_data = extracted
            state.task_completed = True
            state.success = True
            state.navigation_history.append(f"Data extracted: {len(extracted['elements'])} elements")
            logger.info(f"Extraction complete: {len(extracted['elements'])} elements found")
        except Exception as e:
            logger.error(f"Data extraction failed: {e}")
            state.error_message = f"Extraction failed: {str(e)}"
        return self._ensure_state(state, state)


    async def _interact_with_page(self, state: BrowserState) -> BrowserState:
        logger.info("Interacting with page elements...")
        try:
            interaction_results = []
            if state.form_data:
                form_result = await self.browser.fill_form(state.form_data)
                interaction_results.append(f"Form filled: {form_result}")
            for target in state.click_targets:
                if target and isinstance(target, str):
                    click_result = await self.browser.click_element(target, wait_for_navigation=False)
                    interaction_results.append(f"Clicked {target}: {click_result}")
                    await asyncio.sleep(1)
            state.navigation_history.extend(interaction_results)
            await asyncio.sleep(2)  # Let page settle
        except Exception as e:
            logger.error(f"Page interaction failed: {e}")
            state.error_message = f"Interaction failed: {str(e)}"
        return self._ensure_state(state, state)

    async def _search_content(self, state: BrowserState) -> BrowserState:
        logger.info("Searching page content...")
        try:
            search_terms = []
            task_words = state.task_description.lower().split()
            for word in task_words:
                if len(word) > 3 and word not in ['find', 'search', 'look', 'page', 'website']:
                    search_terms.append(word)
            search_results = {}
            for term in search_terms[:5]:
                results = await self.browser.search_text(term)
                if results:
                    search_results[term] = results
            state.extracted_data = {
                "search_results": search_results,
                "search_terms": search_terms,
                "timestamp": datetime.now().isoformat()
            }
            state.task_completed = True
            state.success = True
            state.navigation_history.append(f"Search completed for terms: {search_terms}")
        except Exception as e:
            logger.error(f"Content search failed: {e}")
            state.error_message = f"Search failed: {str(e)}"
        return self._ensure_state(state, state)

    async def _complete_task(self, state: BrowserState) -> BrowserState:
        logger.info("Completing task...")
        state.current_step = "completed"
        state.task_completed = True
        if not state.success:
            state.success = len(state.extracted_data) > 0 or len(state.navigation_history) > 1
        completion_summary = {
            "task": state.task_description,
            "success": state.success,
            "final_url": state.current_url,
            "steps_taken": len(state.navigation_history),
            "data_extracted": bool(state.extracted_data),
            "screenshot": state.screenshot_path
        }
        state.navigation_history.append(f"Task completed: {completion_summary}")
        await self.browser.cleanup()
        return self._ensure_state(state, state)

    async def _handle_error(self, state: BrowserState) -> BrowserState:
        logger.error(f"Handling error: {state.error_message}")
        state.current_step = "error"
        if state.retry_count < state.max_retries and "navigation" in state.error_message.lower():
            state.retry_count += 1
            state.error_message = ""
            state.navigation_history.append(f"Retrying... (attempt {state.retry_count})")
            await asyncio.sleep(2)
        else:
            state.task_completed = True
            state.success = False
            state.navigation_history.append(f"Task failed after {state.retry_count} retries: {state.error_message}")
            await self.browser.cleanup()
        return self._ensure_state(state, state)

    async def execute_task(self, url: str, task: str, task_type: str = "extract", form_data: dict = None) -> dict:
        initial_state = BrowserState(
            target_url=url,
            task_description=task,
            task_type=task_type,
            form_data=form_data or {}
        )
        logger.info(f"Starting task: {task}")
        logger.info(f"Target URL: {url}")
        try:
            final_state = await self.graph.ainvoke(initial_state)
            # Defensive: if final_state is not BrowserState, try to coerce it
            final_state = self._ensure_state(final_state, initial_state)
            result = {
                "success": final_state.success,
                "task": task,
                "url": url,
                "final_url": final_state.current_url,
                "extracted_data": final_state.extracted_data,
                "navigation_history": final_state.navigation_history,
                "screenshot": final_state.screenshot_path,
                "error": final_state.error_message,
                "timestamp": datetime.now().isoformat()
            }
            return result
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            await self.browser.cleanup()
            return {
                "success": False,
                "task": task,
                "url": url,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
