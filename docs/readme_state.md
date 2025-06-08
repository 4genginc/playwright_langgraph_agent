# 🧩 Pedagogical Guide: `state.py`

A **modular, pedagogical walk-through and starter template** for your agent's `state.py`, which defines the agent's state object. This module is critical for orchestrating LLM workflows using LangGraph and ensures that both your agent and browser logic are cleanly decoupled from shared memory/state management.

---

## 📂 Where Does This Fit?

```plaintext
playwright_langgraph_agent/
└── state.py           # This file (shared agent state)
```

* **Role:** Central location for the agent's state dataclass/structure—used by LangGraph, agent nodes, and for logging/history.

---

## 🎯 Purpose and Design Philosophy

* **Single Source of Truth:** All info about the agent's progress, goals, and outputs lives here.
* **Data-Oriented:** Designed as a `@dataclass` (or Pydantic model) for clarity and type safety.
* **Traceable:** History and memory fields for easy debugging and repeatability.
* **Easy to Extend:** Add fields for new features (e.g., file uploads, cookies, custom metadata) without refactoring agent logic.

---

## 🚦 Typical Usage Pattern

```python
from state import BrowserState

# Create new state instance for a task
state = BrowserState(target_url="https://example.com", task_description="Extract links")

# Update as you go (used by agent nodes)
state.current_url = "https://example.com"
state.page_title = "Example Domain"
state.navigation_history.append("Navigated to example.com")
```

* **In practice:** The agent receives a state, updates fields at each node, and returns it (LangGraph manages state transitions).

---

## 🧩 Main Fields – Pedagogical Walkthrough

```python
from dataclasses import dataclass, field
from typing import Dict, Any, List

@dataclass
class BrowserState:
    # Navigation and page info
    current_url: str = ""
    target_url: str = ""
    page_title: str = ""
    page_content: str = ""

    # Task control
    task_description: str = ""
    task_type: str = ""         # e.g. "extract", "interact", "search"
    current_step: str = "initialize"

    # User interaction
    form_data: Dict[str, str] = field(default_factory=dict)
    click_targets: List[str] = field(default_factory=list)
    extracted_data: Dict[str, Any] = field(default_factory=dict)

    # Agent memory/history
    navigation_history: List[str] = field(default_factory=list)
    screenshot_path: str = ""
    page_elements: List[Dict] = field(default_factory=list)

    # Error handling
    error_message: str = ""
    retry_count: int = 0
    max_retries: int = 3

    # Completion status
    task_completed: bool = False
    success: bool = False
```

**Field Purposes:**

* `current_url`, `target_url`, `page_title`, `page_content`: Browser/page tracking.
* `task_description`, `task_type`, `current_step`: Agent's goal, workflow state.
* `form_data`, `click_targets`: Inputs for interactive tasks.
* `extracted_data`: Where all scraped/search results go.
* `navigation_history`: All major actions/steps, useful for debug or audit.
* `screenshot_path`, `page_elements`: Extra outputs for UI or further agent steps.
* `error_message`, `retry_count`, `max_retries`: Robust error/retry support.
* `task_completed`, `success`: Flow control and agent stop conditions.

---

## 🌟 Pedagogical Tips

* **Keep state objects serializable:** Prefer built-in types and clear structures (dicts/lists, not objects).
* **Extend as needed:** Add fields for new agent skills (e.g., authentication state, file uploads, custom logs).
* **Logging:** `navigation_history` and `error_message` make debugging and user reporting much easier.
* **Stateless agent logic:** All node functions operate on and return state, which keeps agents pure and testable.

---

## 📚 Example: Full Minimal File Template

```python
from dataclasses import dataclass, field
from typing import Dict, Any, List

@dataclass
class BrowserState:
    current_url: str = ""
    target_url: str = ""
    page_title: str = ""
    page_content: str = ""
    task_description: str = ""
    task_type: str = ""
    current_step: str = "initialize"
    form_data: Dict[str, str] = field(default_factory=dict)
    click_targets: List[str] = field(default_factory=list)
    extracted_data: Dict[str, Any] = field(default_factory=dict)
    navigation_history: List[str] = field(default_factory=list)
    screenshot_path: str = ""
    page_elements: List[Dict] = field(default_factory=list)
    error_message: str = ""
    retry_count: int = 0
    max_retries: int = 3
    task_completed: bool = False
    success: bool = False
```

---

## 🔬 Extension Ideas

* Add timestamps, session IDs, authentication/cookie state.
* For multi-turn or multi-agent: add fields for agent identity, sub-task memory.
* For multi-page or multi-browser: make a list of states or include browser IDs.

---

## 🤝 Attributions

* Python stdlib `dataclasses` and best practices for agent memory/state.
* Inspired by LangGraph and practical LLM workflow patterns.
