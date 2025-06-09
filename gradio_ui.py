#!/usr/bin/env python3
"""
gradio_ui.py

Advanced Gradio interface for the Playwright LangGraph Agent
Provides a comprehensive web UI for web browsing automation tasks.
"""

import gradio as gr
import asyncio
import json
import os
import logging
import traceback
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd

# Import your agent components
from config import load_env, get_api_key, setup_logging
from agent.web_browsing_agent import WebBrowsingAgent
from toolkit.web_toolkit import export_json, export_csv, run_batch
from utils import ensure_dir, timestamp_str, save_json

# Load environment and setup logging
load_env()
setup_logging("INFO")
logger = logging.getLogger(__name__)

# Ensure results directory exists
ensure_dir("results/gradio_sessions")
ensure_dir("results/batch_runs")
ensure_dir("results/screenshots")

class GradioAgentInterface:
    """Main interface class for the Gradio web UI"""
    
    def __init__(self):
        self.agent = None
        self.session_results = []
        self.batch_results = []
        
    def initialize_agent(self, api_key: str, headless: bool = True) -> tuple:
        """Initialize the agent with API key"""
        try:
            if not api_key or not api_key.strip():
                return "❌ Error: Please provide an OpenAI API key", "No API Key"
            
            # Validate API key format
            if not api_key.startswith('sk-'):
                return "❌ Error: Invalid API key format (should start with 'sk-')", "Invalid API Key"
            
            # Initialize the agent
            self.agent = WebBrowsingAgent(api_key, headless=headless)
            
            # Test the agent by creating a simple instance
            logger.info("Agent initialized successfully")
            return "✅ Agent initialized successfully!", "Agent Ready ✅"
            
        except Exception as e:
            error_msg = f"❌ Failed to initialize agent: {str(e)}"
            logger.error(error_msg)
            return error_msg, "Initialization Failed ❌"
    
    async def execute_single_task(
        self, 
        url: str, 
        task: str, 
        task_type: str, 
        form_data_json: str = "",
        api_key: str = ""
    ) -> tuple:
        """Execute a single web browsing task"""
        try:
            # Validate inputs
            if not url or not url.strip():
                return "❌ Error: Please provide a valid URL", "", "", None
            
            if not task or not task.strip():
                return "❌ Error: Please provide a task description", "", "", None
            
            # Validate URL format
            url = url.strip()
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            # Initialize agent if needed
            if not self.agent:
                init_result, _ = self.initialize_agent(api_key, headless=True)
                if "Error" in init_result:
                    return init_result, "", "", None
            
            # Parse form data if provided
            form_data = {}
            if form_data_json.strip():
                try:
                    form_data = json.loads(form_data_json)
                except json.JSONDecodeError:
                    return "❌ Invalid JSON in form data", "", "", None
            
            # Execute the task
            result = await self.agent.execute_task(url, task, task_type, form_data)
            
            # Store result
            self.session_results.append(result)
            
            # Format output
            status = "✅ Success" if result.get("success") else "❌ Failed"
            
            # Create detailed result display
            result_display = self._format_result_display(result)
            
            # Create navigation history
            nav_history = "\n".join([
                f"{i+1}. {step}" 
                for i, step in enumerate(result.get("navigation_history", []))
            ])
            
            # Handle screenshot - ensure it's a valid file path
            screenshot_path = result.get("screenshot", "")
            if screenshot_path and os.path.exists(screenshot_path) and os.path.isfile(screenshot_path):
                screenshot_display = screenshot_path
            else:
                screenshot_display = None
            
            return status, result_display, nav_history, screenshot_display
            
        except Exception as e:
            error_msg = f"❌ Task execution failed: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            return error_msg, "", "", None
    
    def _format_result_display(self, result: Dict[str, Any]) -> str:
        """Format result for display in the UI"""
        lines = []
        lines.append("=== TASK RESULT ===")
        lines.append(f"🎯 Task: {result.get('task', 'N/A')}")
        lines.append(f"🌐 URL: {result.get('url', 'N/A')}")
        lines.append(f"📍 Final URL: {result.get('final_url', 'N/A')}")
        lines.append(f"✅ Success: {result.get('success', False)}")
        lines.append(f"⏰ Timestamp: {result.get('timestamp', 'N/A')}")
        
        if result.get('error'):
            lines.append(f"❌ Error: {result['error']}")
        
        # Extract data summary
        extracted_data = result.get('extracted_data', {})
        if extracted_data:
            lines.append("\n=== EXTRACTED DATA ===")
            if 'elements' in extracted_data:
                elements = extracted_data['elements']
                lines.append(f"📊 Elements found: {len(elements)}")
                for i, elem in enumerate(elements[:5]):  # Show first 5
                    text = elem.get('text', '')[:100]
                    lines.append(f"  {i+1}. {elem.get('tag', 'unknown')}: {text}...")
                if len(elements) > 5:
                    lines.append(f"  ... and {len(elements) - 5} more elements")
            
            if 'tables' in extracted_data:
                lines.append(f"📋 Tables found: {len(extracted_data['tables'])}")
            
            if 'search_results' in extracted_data:
                search_results = extracted_data['search_results']
                lines.append(f"🔍 Search results: {len(search_results)} terms")
        
        return "\n".join(lines)
    
    async def execute_batch_tasks(
        self, 
        batch_data: str, 
        max_concurrent: int,
        api_key: str
    ) -> tuple:
        """Execute batch tasks from CSV/JSON data"""
        try:
            # Initialize agent if needed
            if not self.agent:
                init_result, _ = self.initialize_agent(api_key, headless=True)
                if "Error" in init_result:
                    return init_result, "", ""
            
            # Parse batch data
            try:
                tasks = json.loads(batch_data)
                if not isinstance(tasks, list):
                    return "❌ Batch data must be a JSON array", "", ""
            except json.JSONDecodeError:
                return "❌ Invalid JSON in batch data", "", ""
            
            # Validate tasks
            for i, task in enumerate(tasks):
                if not all(key in task for key in ['url', 'task']):
                    return f"❌ Task {i+1} missing required fields (url, task)", "", ""
            
            # Execute batch
            results = await run_batch(self.agent, tasks, max_concurrent)
            self.batch_results.extend(results)
            
            # Save results
            timestamp = timestamp_str()
            batch_dir = f"results/batch_runs/batch_{timestamp}"
            ensure_dir(batch_dir)
            
            # Save individual results
            export_json(results, f"{batch_dir}/results.json")
            export_csv(results, f"{batch_dir}/results.csv")
            
            # Create summary
            successful = sum(1 for r in results if r.get('success'))
            summary = f"""
=== BATCH EXECUTION COMPLETE ===
📊 Total tasks: {len(results)}
✅ Successful: {successful}
❌ Failed: {len(results) - successful}
💾 Results saved to: {batch_dir}
⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """.strip()
            
            # Create detailed results
            detailed_results = []
            for i, result in enumerate(results):
                status = "✅" if result.get('success') else "❌"
                detailed_results.append(
                    f"{i+1}. {status} {result.get('url', 'N/A')} - {result.get('task', 'N/A')[:50]}..."
                )
            
            detailed_output = "\n".join(detailed_results)
            
            return summary, detailed_output, batch_dir
            
        except Exception as e:
            error_msg = f"❌ Batch execution failed: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            return error_msg, "", ""
    
    def get_session_history(self) -> str:
        """Get formatted session history"""
        if not self.session_results:
            return "No tasks executed in this session."
        
        lines = ["=== SESSION HISTORY ==="]
        for i, result in enumerate(self.session_results):
            status = "✅" if result.get('success') else "❌"
            timestamp = result.get('timestamp', 'N/A')
            url = result.get('url', 'N/A')
            task = result.get('task', 'N/A')
            lines.append(f"{i+1}. {status} [{timestamp}] {url} - {task[:50]}...")
        
        return "\n".join(lines)
    
    def export_session_results(self, format_type: str) -> tuple:
        """Export session results"""
        try:
            if not self.session_results:
                return "❌ No results to export", ""
            
            timestamp = timestamp_str()
            filename = f"results/gradio_sessions/session_{timestamp}"
            
            if format_type == "JSON":
                filename += ".json"
                export_json(self.session_results, filename)
            else:  # CSV
                filename += ".csv"
                export_csv(self.session_results, filename)
            
            return f"✅ Results exported to: {filename}", filename
            
        except Exception as e:
            return f"❌ Export failed: {str(e)}", ""
    
    def clear_session(self) -> str:
        """Clear current session results"""
        self.session_results = []
        return "✅ Session cleared"

# Initialize the interface
interface = GradioAgentInterface()

def create_gradio_interface():
    """Create and configure the Gradio interface"""
    
    with gr.Blocks(
        title="🕷️ Playwright LangGraph Agent",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .tab-nav {
            font-weight: bold;
        }
        """
    ) as demo:
        
        gr.Markdown("""
        # 🕷️ Playwright LangGraph Agent
        
        **Autonomous web browsing with LLM-powered decision making**
        
        This interface allows you to automate web browsing tasks using AI. The agent can:
        - 📄 Extract data from web pages
        - 🔗 Interact with forms and buttons  
        - 🔍 Search for specific content
        - 📊 Process multiple URLs in batch
        """)
        
        # API Key input (shared across tabs)
        with gr.Row():
            with gr.Column(scale=3):
                api_key_input = gr.Textbox(
                    label="🔑 OpenAI API Key",
                    placeholder="sk-...",
                    type="password",
                    value=os.getenv("OPENAI_API_KEY", "")
                )
            with gr.Column(scale=2):
                agent_status = gr.Textbox(
                    label="Agent Status",
                    value="Not Initialized",
                    interactive=False
                )
            with gr.Column(scale=1):
                init_agent_btn = gr.Button("🔄 Initialize Agent", variant="secondary")
        
        with gr.Tabs():
            # Single Task Tab
            with gr.Tab("🎯 Single Task", elem_classes=["tab-nav"]):
                with gr.Row():
                    with gr.Column(scale=2):
                        url_input = gr.Textbox(
                            label="🌐 Target URL",
                            placeholder="https://example.com",
                            lines=1
                        )
                        
                        task_input = gr.Textbox(
                            label="📝 Task Description",
                            placeholder="Extract the main headline and article links",
                            lines=2
                        )
                        
                        task_type = gr.Dropdown(
                            label="🔧 Task Type",
                            choices=["extract", "interact", "search"],
                            value="extract"
                        )
                        
                        form_data_input = gr.Textbox(
                            label="📋 Form Data (JSON)",
                            placeholder='{"#username": "test", "#password": "demo"}',
                            lines=3
                        )
                        
                        execute_btn = gr.Button("🚀 Execute Task", variant="primary")
                    
                    with gr.Column(scale=3):
                        task_status = gr.Textbox(
                            label="📊 Task Status",
                            lines=2,
                            interactive=False
                        )
                        
                        result_display = gr.Textbox(
                            label="📄 Results",
                            lines=10,
                            interactive=False
                        )
                        
                        nav_history = gr.Textbox(
                            label="🧭 Navigation History",
                            lines=5,
                            interactive=False
                        )
                
                # Screenshot display
                screenshot_display = gr.Image(
                    label="📸 Screenshot",
                    height=300
                )
            
            # Batch Processing Tab
            with gr.Tab("📊 Batch Processing", elem_classes=["tab-nav"]):
                gr.Markdown("""
                ### Batch Task Format
                Provide a JSON array of tasks. Each task should have:
                - `url`: Target URL
                - `task`: Task description
                - `task_type`: "extract", "interact", or "search" (optional, defaults to "extract")
                - `form_data`: Form data object (optional)
                """)
                
                with gr.Row():
                    with gr.Column():
                        batch_data_input = gr.Textbox(
                            label="📋 Batch Tasks (JSON)",
                            placeholder='''[
  {
    "url": "https://news.ycombinator.com",
    "task": "Extract top 5 headlines",
    "task_type": "extract"
  },
  {
    "url": "https://example.com",
    "task": "Get page title and main content",
    "task_type": "extract"
  }
]''',
                            lines=10
                        )
                        
                        max_concurrent = gr.Slider(
                            label="🔄 Max Concurrent Tasks",
                            minimum=1,
                            maximum=10,
                            value=3,
                            step=1
                        )
                        
                        batch_execute_btn = gr.Button("🚀 Execute Batch", variant="primary")
                    
                    with gr.Column():
                        batch_status = gr.Textbox(
                            label="📊 Batch Status",
                            lines=8,
                            interactive=False
                        )
                        
                        batch_results_display = gr.Textbox(
                            label="📄 Batch Results Summary",
                            lines=10,
                            interactive=False
                        )
                        
                        batch_output_dir = gr.Textbox(
                            label="💾 Output Directory",
                            interactive=False
                        )
            
            # Session Management Tab
            with gr.Tab("💾 Session Management", elem_classes=["tab-nav"]):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 📈 Session History")
                        
                        session_history_display = gr.Textbox(
                            label="📋 Current Session Results",
                            lines=15,
                            interactive=False
                        )
                        
                        with gr.Row():
                            refresh_history_btn = gr.Button("🔄 Refresh History")
                            clear_session_btn = gr.Button("🗑️ Clear Session", variant="secondary")
                    
                    with gr.Column():
                        gr.Markdown("### 💾 Export Options")
                        
                        export_format = gr.Dropdown(
                            label="📁 Export Format",
                            choices=["JSON", "CSV"],
                            value="JSON"
                        )
                        
                        export_btn = gr.Button("📤 Export Results", variant="primary")
                        
                        export_status = gr.Textbox(
                            label="📊 Export Status",
                            lines=3,
                            interactive=False
                        )
                        
                        export_file_path = gr.Textbox(
                            label="📁 Export File Path",
                            interactive=False
                        )
            
            # Help & Examples Tab
            with gr.Tab("❓ Help & Examples", elem_classes=["tab-nav"]):
                gr.Markdown("""
                ## 📚 Usage Guide
                
                ### 🎯 Single Tasks
                
                **Extract Data:**
                - URL: `https://news.ycombinator.com`
                - Task: `Extract the top 10 news headlines and their links`
                - Type: `extract`
                
                **Interact with Forms:**
                - URL: `https://httpbin.org/forms/post`
                - Task: `Fill out the contact form`
                - Type: `interact`
                - Form Data: `{"input[name='custname']": "John Doe", "input[name='custemail']": "john@example.com"}`
                
                **Search Content:**
                - URL: `https://example.com`
                - Task: `Find information about pricing`
                - Type: `search`
                
                ### 📊 Batch Processing
                
                ```json
                [
                  {
                    "url": "https://news.ycombinator.com",
                    "task": "Extract top 5 headlines",
                    "task_type": "extract"
                  },
                  {
                    "url": "https://httpbin.org/html",
                    "task": "Get page content",
                    "task_type": "extract"
                  }
                ]
                ```
                
                ### 🔧 Form Data Format
                
                Form data should be a JSON object where keys are CSS selectors and values are the input values:
                
                ```json
                {
                  "#username": "myuser",
                  "input[name='password']": "mypass",
                  ".email-field": "user@example.com"
                }
                ```
                
                ### 💡 Tips
                
                - 🔍 Use specific, clear task descriptions
                - 🎯 Test with simple pages first
                - 📸 Screenshots are automatically captured
                - 🔄 The agent will retry failed operations
                - 💾 All results are automatically saved
                """)
        
        # Event handlers
        def execute_single_task_sync(url, task, task_type, form_data, api_key):
            return asyncio.run(interface.execute_single_task(url, task, task_type, form_data, api_key))
        
        def execute_batch_sync(batch_data, max_concurrent, api_key):
            return asyncio.run(interface.execute_batch_tasks(batch_data, max_concurrent, api_key))
        
        # Single task execution
        execute_btn.click(
            fn=execute_single_task_sync,
            inputs=[url_input, task_input, task_type, form_data_input, api_key_input],
            outputs=[task_status, result_display, nav_history, screenshot_display]
        )
        
        # Batch execution
        batch_execute_btn.click(
            fn=execute_batch_sync,
            inputs=[batch_data_input, max_concurrent, api_key_input],
            outputs=[batch_status, batch_results_display, batch_output_dir]
        )
        
        # Session management
        refresh_history_btn.click(
            fn=interface.get_session_history,
            outputs=[session_history_display]
        )
        
        clear_session_btn.click(
            fn=interface.clear_session,
            outputs=[session_history_display]
        )
        
        export_btn.click(
            fn=interface.export_session_results,
            inputs=[export_format],
            outputs=[export_status, export_file_path]
        )
        
        # Initialize agent when API key is provided
        def init_agent_sync(api_key):
            if api_key and api_key.strip():
                result, status = interface.initialize_agent(api_key)
                return status
            else:
                return "Enter API Key to Initialize"
        
        # Auto-initialize on startup if API key exists in environment
        def auto_initialize():
            api_key = os.getenv("OPENAI_API_KEY", "")
            if api_key:
                result, status = interface.initialize_agent(api_key)
                return api_key, status
            return "", "Enter API Key to Initialize"
        
        # Auto-initialize when interface loads
        demo.load(
            fn=auto_initialize,
            outputs=[api_key_input, agent_status]
        )
        
        # Manual initialization button
        init_agent_btn.click(
            fn=init_agent_sync,
            inputs=[api_key_input],
            outputs=[agent_status]
        )
        
        # Re-initialize when API key changes
        api_key_input.change(
            fn=init_agent_sync,
            inputs=[api_key_input],
            outputs=[agent_status]
        )
    
    return demo

def main():
    """Launch the Gradio interface - for direct execution"""
    try:
        # Create and launch the interface
        demo = create_gradio_interface()
        
        # Find available port
        import socket
        port = 7860
        for test_port in range(7860, 7870):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('localhost', test_port))
                    port = test_port
                    break
            except OSError:
                continue
        
        # Launch with custom settings
        demo.launch(
            server_name="0.0.0.0",  # Allow external access
            server_port=port,
            share=False,  # Set to True to create a public link
            debug=True,
            show_error=True,
            quiet=False
        )
        
    except Exception as e:
        logger.error(f"Failed to launch Gradio interface: {e}")
        print(f"❌ Error launching interface: {e}")

if __name__ == "__main__":
    main()
