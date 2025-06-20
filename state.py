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
    task_type: str = ""  # e.g. "extract", "interact", "search"
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

    # Step counting for loop prevention
    step_count: int = 0