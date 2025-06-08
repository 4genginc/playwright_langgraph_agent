# utils.py

import os
import sys
import logging
import json
from datetime import datetime

def ensure_dir(path):
    """
    Ensure that a directory exists.
    """
    os.makedirs(path, exist_ok=True)

def timestamp_str(fmt="%Y%m%d_%H%M%S"):
    """
    Return a timestamp string for filenames/logs.
    """
    return datetime.now().strftime(fmt)

def save_json(data, filename):
    """
    Save a dict/list as pretty JSON.
    """
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_json(filename):
    """
    Load and return JSON from a file.
    """
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

def setup_basic_logging(level="INFO", log_file=None):
    """
    Simple logging config (fallback/standalone).
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

def print_banner(msg, char="=", width=60):
    """
    Print a banner message for CLI UX.
    """
    print(char * width)
    print(msg)
    print(char * width)

def die(msg, code=1):
    """
    Print an error and exit.
    """
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)

# You can extend this with more helpers as you need!
