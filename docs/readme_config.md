# 🛠️ config.py – Starter Template & Usage Guide

Centralizes all environment, API key, and logging configuration for your agent project. Import and use in `main.py` and other entry scripts.

---

```python
# config.py

import os
import logging
from dotenv import load_dotenv

def load_env(dotenv_path=".env"):
    """
    Load environment variables from .env file.
    Call this at the start of your main.py.
    """
    load_dotenv(dotenv_path, override=True)

def get_api_key(var="OPENAI_API_KEY"):
    """
    Get the OpenAI API key or other required key.
    """
    key = os.getenv(var)
    if not key:
        raise EnvironmentError(f"Missing environment variable: {var}")
    return key

def setup_logging(level="INFO", log_file=None):
    """
    Set up logging for the project.
    Usage: setup_logging("DEBUG")
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers
    )

# Example: add custom config helper

def get_default_model():
    return os.getenv("DEFAULT_MODEL", "gpt-4o")
```

---

## 🧑‍💻 Usage Example (in main.py)

```python
from config import load_env, get_api_key, setup_logging

load_env()
setup_logging("INFO")
api_key = get_api_key()
```

---

## 🔬 Test Example (pytest style)

```python
def test_env_loading(monkeypatch):
    import config
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key")
    assert config.get_api_key() == "dummy-key"
```
