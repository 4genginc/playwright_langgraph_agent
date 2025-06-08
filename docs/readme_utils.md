# 🛠️ `utils.py` — Complete Pedagogical Guide & Usage Examples

This module provides **essential utility functions** that simplify common tasks throughout your Playwright-LangGraph agent project. These utilities help maintain clean, DRY (Don't Repeat Yourself) code and provide consistent patterns for file operations, logging, and user feedback.

---

## 📋 Complete Function Reference

| Function                               | Purpose                                                   | Return Type |
| -------------------------------------- | --------------------------------------------------------- | ----------- |
| `ensure_dir(path)`                     | Create directory if it doesn't exist (safe, idempotent)  | `None`      |
| `timestamp_str(fmt="%Y%m%d_%H%M%S")`   | Generate timestamp string for filenames/logs             | `str`       |
| `save_json(data, filename)`            | Save Python data as pretty, readable JSON                | `None`      |
| `load_json(filename)`                  | Load JSON from disk into Python dict/list                | `dict/list` |
| `setup_basic_logging(level, log_file)` | Quick/portable logging setup for scripts                 | `None`      |
| `print_banner(msg, char="=", width=60)` | Print emphasized banner for CLI feedback               | `None`      |
| `die(msg, code=1)`                     | Print error to stderr and exit script (fail fast)       | `Never returns` |

---

## 🚀 Practical Usage Examples

### 1. Directory and File Management

```python
from utils import ensure_dir, save_json, load_json, timestamp_str

# Create results directory safely
ensure_dir("results/screenshots")
ensure_dir("data/exports")

# Generate timestamped filenames
timestamp = timestamp_str()  # "20241208_143052"
screenshot_file = f"results/screenshots/capture_{timestamp}.png"

# Save agent results as JSON
agent_results = {
    "url": "https://example.com",
    "extracted_data": {"title": "Example", "links": []},
    "timestamp": timestamp
}
save_json(agent_results, f"results/session_{timestamp}.json")

# Load previous results
try:
    previous_results = load_json("results/session_20241207_120000.json")
    print(f"Loaded {len(previous_results)} previous results")
except FileNotFoundError:
    print("No previous results found")
```

### 2. Logging Setup for Scripts

```python
from utils import setup_basic_logging, print_banner
import logging

# Set up logging for a batch script
setup_basic_logging("DEBUG", "batch_processing.log")
logger = logging.getLogger(__name__)

print_banner("Starting Batch Web Scraping")
logger.info("Batch processing started")

# Your agent code here...
logger.debug("Processing URL: https://example.com")
logger.warning("Rate limit approached")
logger.error("Failed to extract data from malformed page")
```

### 3. Error Handling and Script Exit

```python
from utils import die, print_banner
import os

def main():
    print_banner("Web Agent Startup Check")
    
    # Check for required environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        die("OPENAI_API_KEY environment variable is required!")
    
    # Check for required directories
    if not os.path.exists("config"):
        die("Config directory not found. Run setup first.")
    
    print("✅ All checks passed, starting agent...")

if __name__ == "__main__":
    main()
```

### 4. Agent Integration Examples

```python
from utils import ensure_dir, save_json, timestamp_str, print_banner
from agent.web_browsing_agent import WebBrowsingAgent
import asyncio

async def run_batch_extraction():
    print_banner("Batch News Extraction", char="🚀", width=50)
    
    # Prepare output directory
    ensure_dir("batch_results")
    
    # Initialize agent
    agent = WebBrowsingAgent(api_key="your-key", headless=True)
    
    urls = [
        "https://news.ycombinator.com",
        "https://techcrunch.com",
        "https://www.reddit.com/r/technology"
    ]
    
    all_results = []
    timestamp = timestamp_str()
    
    for i, url in enumerate(urls, 1):
        print(f"Processing {i}/{len(urls)}: {url}")
        
        result = await agent.execute_task(
            url=url,
            task="Extract top 5 headlines",
            task_type="extract"
        )
        
        # Add metadata
        result["batch_id"] = timestamp
        result["sequence"] = i
        all_results.append(result)
        
        # Save individual result
        save_json(result, f"batch_results/result_{timestamp}_{i:02d}.json")
    
    # Save consolidated results
    save_json(all_results, f"batch_results/batch_{timestamp}_complete.json")
    print_banner(f"✅ Batch complete! {len(all_results)} results saved")

# Run the batch
asyncio.run(run_batch_extraction())
```

### 5. Configuration and Setup Scripts

```python
#!/usr/bin/env python3
"""Setup script for the agent environment"""

from utils import ensure_dir, save_json, print_banner, die
import os

def setup_environment():
    print_banner("🔧 Agent Environment Setup")
    
    # Create required directories
    directories = [
        "results",
        "results/screenshots", 
        "results/exports",
        "logs",
        "config",
        "data"
    ]
    
    for directory in directories:
        ensure_dir(directory)
        print(f"✅ Created: {directory}")
    
    # Create default configuration
    default_config = {
        "agent": {
            "headless": True,
            "viewport_width": 1280,
            "viewport_height": 720,
            "max_retries": 3
        },
        "browser": {
            "timeout": 30000,
            "wait_until": "networkidle"
        },
        "export": {
            "format": "json",
            "include_screenshots": True
        }
    }
    
    save_json(default_config, "config/default.json")
    print("✅ Created: config/default.json")
    
    # Check environment variables
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        die(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    print_banner("🎉 Setup Complete!")

if __name__ == "__main__":
    setup_environment()
```

### 6. Data Analysis and Reporting

```python
from utils import load_json, save_json, print_banner, timestamp_str
import glob
from collections import Counter

def analyze_batch_results():
    print_banner("📊 Batch Results Analysis")
    
    # Load all result files
    result_files = glob.glob("batch_results/result_*.json")
    all_results = []
    
    for file_path in result_files:
        try:
            result = load_json(file_path)
            all_results.append(result)
        except Exception as e:
            print(f"⚠️  Failed to load {file_path}: {e}")
    
    # Analyze results
    success_count = sum(1 for r in all_results if r.get("success"))
    failure_count = len(all_results) - success_count
    
    # Count error types
    error_types = Counter()
    for result in all_results:
        if not result.get("success") and result.get("error"):
            error_types[result["error"]] += 1
    
    # Generate report
    report = {
        "analysis_timestamp": timestamp_str(),
        "total_results": len(all_results),
        "successful": success_count,
        "failed": failure_count,
        "success_rate": success_count / len(all_results) if all_results else 0,
        "common_errors": dict(error_types.most_common(5)),
        "processed_files": len(result_files)
    }
    
    # Save report
    save_json(report, f"batch_results/analysis_{timestamp_str()}.json")
    
    # Print summary
    print(f"📈 Total Results: {report['total_results']}")
    print(f"✅ Successful: {report['successful']}")
    print(f"❌ Failed: {report['failed']}")
    print(f"📊 Success Rate: {report['success_rate']:.1%}")
    
    if error_types:
        print("\n🔍 Common Errors:")
        for error, count in error_types.most_common(3):
            print(f"  • {error}: {count} times")

# Run analysis
analyze_batch_results()
```

---

## 🎯 Best Practices & Patterns

### 1. **Always Use `ensure_dir()` Before File Operations**
```python
# ✅ Good
ensure_dir("results/screenshots")
screenshot_path = "results/screenshots/capture.png"

# ❌ Avoid - might fail if directory doesn't exist
screenshot_path = "results/screenshots/capture.png"  # Could crash
```

### 2. **Timestamp Everything for Traceability**
```python
# ✅ Good - unique, sortable filenames
timestamp = timestamp_str()
save_json(data, f"results/session_{timestamp}.json")

# ❌ Avoid - files get overwritten
save_json(data, "results/session.json")
```

### 3. **Use `die()` for Critical Failures**
```python
# ✅ Good - fail fast with clear message
if not api_key:
    die("OPENAI_API_KEY is required but not set!")

# ❌ Avoid - unclear what went wrong
if not api_key:
    exit(1)
```

### 4. **Combine Utilities for Common Patterns**
```python
def save_agent_result(result, session_name):
    """Save agent result with timestamp and proper directory structure"""
    ensure_dir(f"results/{session_name}")
    timestamp = timestamp_str()
    filename = f"results/{session_name}/result_{timestamp}.json"
    save_json(result, filename)
    return filename
```

---

## 🔬 Extension Ideas

### Add More Utilities As Needed:

```python
def validate_url(url):
    """Validate URL format"""
    import re
    pattern = re.compile(r'^https?://.+')
    return bool(pattern.match(url))

def human_readable_size(bytes_size):
    """Convert bytes to human readable size"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"

def retry_on_failure(max_attempts=3, delay=1):
    """Decorator for retrying failed operations"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(delay)
        return wrapper
    return decorator
```

---

## 🧪 Testing Your Utils

```python
# test_utils.py
import tempfile
import pytest
from utils import ensure_dir, save_json, load_json, timestamp_str

def test_ensure_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = f"{tmpdir}/new/nested/directory"
        ensure_dir(test_path)
        assert os.path.exists(test_path)

def test_json_roundtrip():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        test_data = {"key": "value", "number": 42}
        save_json(test_data, f.name)
        loaded_data = load_json(f.name)
        assert loaded_data == test_data

def test_timestamp_format():
    timestamp = timestamp_str()
    assert len(timestamp) == 15  # YYYYMMDD_HHMMSS
    assert '_' in timestamp
```

---

## 💡 Integration with Main Project

Import utilities throughout your project:

```python
# In agent/web_browsing_agent.py
from utils import ensure_dir, timestamp_str

class WebBrowsingAgent:
    async def take_screenshot(self):
        ensure_dir("screenshots")
        path = f"screenshots/capture_{timestamp_str()}.png"
        return await self.browser.take_screenshot(path)

# In main.py
from utils import print_banner, die
import os

def main():
    print_banner("🦜 Playwright LangGraph Agent")
    
    if not os.getenv("OPENAI_API_KEY"):
        die("Please set OPENAI_API_KEY environment variable")
```

---

**Keep your main code clean and focused—let `utils.py` handle the boring stuff!** 🚀