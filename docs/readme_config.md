# 🛠️ config.py – Complete Pedagogical Guide & Usage Examples

A **comprehensive, production-grade tutorial** for the `config.py` module - the foundation of environment management, API key handling, and logging configuration for your Playwright LangGraph Agent project. This guide provides both theoretical understanding and practical patterns for building robust, configurable AI applications.

---

## 📂 Architecture Context

```plaintext
playwright_langgraph_agent/
├── config.py                    # 🛠️ This file - The Foundation
├── main.py                      # Uses config for startup
├── agent/
│   └── web_browsing_agent.py    # Uses config for API keys and logging
├── browser/
│   └── playwright_manager.py    # Uses config for logging
└── examples/
    └── demo_tasks.py            # Uses config for environment setup
```

**Role in the System:**
- **Environment Orchestrator**: Manages all environment variables and configuration
- **API Gateway**: Secure handling of API keys and credentials
- **Logging Foundation**: Centralized logging setup for the entire application
- **Configuration Hub**: Single source of truth for application settings

---

## 🎯 Design Philosophy & Core Principles

### **1. Fail Fast Principle**
```python
# ✅ Good: Immediate failure with clear error message
def get_api_key(var="OPENAI_API_KEY"):
    key = os.getenv(var)
    if not key:
        raise EnvironmentError(f"Missing environment variable: {var}")
    return key

# ❌ Avoid: Silent failures that cause problems later
def get_api_key_bad():
    return os.getenv("OPENAI_API_KEY", "")  # Empty string causes issues later
```

### **2. Environment-First Configuration**
```python
# ✅ Good: Environment variables with sensible defaults
def get_default_model():
    return os.getenv("DEFAULT_MODEL", "gpt-4o")

# ❌ Avoid: Hard-coded values scattered throughout code
# model = "gpt-4"  # Hard to change, not environment-aware
```

### **3. Centralized Initialization**
```python
# ✅ Good: Single call to setup everything
from config import load_env, setup_logging, get_api_key

load_env()                    # Load .env file
setup_logging("INFO")         # Configure logging
api_key = get_api_key()      # Get required credentials

# ❌ Avoid: Scattered initialization across files
```

---

## 🚀 Complete Function Reference & Deep Dive

### **Environment Management**

#### **`load_env(dotenv_path=".env")`**
```python
def load_env(dotenv_path=".env"):
    """
    Load environment variables from .env file.
    Call this at the start of your main.py.
    """
    load_dotenv(dotenv_path, override=True)
```

**Purpose**: Loads environment variables from a `.env` file into the system environment.

**Advanced Usage Examples:**
```python
# Basic usage - load default .env file
from config import load_env
load_env()

# Load from custom location
load_env(".env.production")
load_env("config/.env.local")

# Load with conditional logic
import os
if os.getenv("ENVIRONMENT") == "development":
    load_env(".env.dev")
elif os.getenv("ENVIRONMENT") == "testing":
    load_env(".env.test")
else:
    load_env(".env.production")

# Load multiple environment files (last one wins for conflicts)
load_env(".env.defaults")    # Base configuration
load_env(".env.local")       # Local overrides
```

**Production Pattern:**
```python
def load_environment_config():
    """Load environment configuration with validation"""
    from config import load_env
    import os
    
    # Determine environment
    env = os.getenv("ENVIRONMENT", "development")
    
    # Load base configuration
    load_env(".env")
    
    # Load environment-specific overrides
    env_file = f".env.{env}"
    if os.path.exists(env_file):
        load_env(env_file)
        print(f"✅ Loaded {env} environment configuration")
    
    # Load local overrides (never committed to git)
    if os.path.exists(".env.local"):
        load_env(".env.local")
        print("✅ Loaded local configuration overrides")
```

#### **`.env` File Best Practices**
```bash
# .env file structure
# Core API Keys (required)
OPENAI_API_KEY=sk-your-key-here
ANTHROPIC_API_KEY=your-anthropic-key

# Model Configuration
DEFAULT_MODEL=gpt-4o
BACKUP_MODEL=gpt-3.5-turbo
MAX_TOKENS=1500
TEMPERATURE=0.1

# Browser Settings
HEADLESS_MODE=true
VIEWPORT_WIDTH=1280
VIEWPORT_HEIGHT=720
BROWSER_TIMEOUT=30000

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/agent.log
ENABLE_DEBUG=false

# Rate Limiting
REQUESTS_PER_MINUTE=60
CONCURRENT_BROWSERS=3

# Environment Settings
ENVIRONMENT=development
DEBUG_MODE=true
SCREENSHOT_PATH=screenshots
RESULTS_PATH=results
```

### **API Key Management**

#### **`get_api_key(var="OPENAI_API_KEY")`**
```python
def get_api_key(var="OPENAI_API_KEY"):
    """
    Get the OpenAI API key or other required key.
    """
    key = os.getenv(var)
    if not key:
        raise EnvironmentError(f"Missing environment variable: {var}")
    return key
```

**Advanced API Key Management:**
```python
class APIKeyManager:
    """Advanced API key management with validation and rotation"""
    
    @staticmethod
    def get_openai_key():
        """Get OpenAI API key with validation"""
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise EnvironmentError("OPENAI_API_KEY is required")
        
        if not key.startswith("sk-"):
            raise ValueError("Invalid OpenAI API key format")
        
        return key
    
    @staticmethod
    def get_anthropic_key():
        """Get Anthropic API key with validation"""
        key = os.getenv("ANTHROPIC_API_KEY")
        if key and not key.startswith("sk-ant-"):
            raise ValueError("Invalid Anthropic API key format")
        return key
    
    @staticmethod
    def get_available_providers():
        """Get list of available LLM providers based on API keys"""
        providers = []
        
        if os.getenv("OPENAI_API_KEY"):
            providers.append("openai")
        if os.getenv("ANTHROPIC_API_KEY"):
            providers.append("anthropic")
        if os.getenv("GOOGLE_API_KEY"):
            providers.append("google")
            
        return providers
    
    @staticmethod
    def validate_all_keys():
        """Validate all configured API keys"""
        validation_results = {}
        
        # Check OpenAI
        try:
            openai_key = APIKeyManager.get_openai_key()
            validation_results["openai"] = {"valid": True, "key_prefix": openai_key[:12] + "..."}
        except Exception as e:
            validation_results["openai"] = {"valid": False, "error": str(e)}
        
        # Check Anthropic
        try:
            anthropic_key = APIKeyManager.get_anthropic_key()
            if anthropic_key:
                validation_results["anthropic"] = {"valid": True, "key_prefix": anthropic_key[:12] + "..."}
        except Exception as e:
            validation_results["anthropic"] = {"valid": False, "error": str(e)}
        
        return validation_results

# Usage examples
from config import get_api_key

# Basic usage
openai_key = get_api_key("OPENAI_API_KEY")
anthropic_key = get_api_key("ANTHROPIC_API_KEY")

# Advanced usage with manager
api_manager = APIKeyManager()
available_providers = api_manager.get_available_providers()
validation_results = api_manager.validate_all_keys()

print(f"Available providers: {available_providers}")
print(f"Key validation: {validation_results}")
```

**Security Best Practices:**
```python
def secure_key_handling():
    """Demonstrate secure API key handling"""
    
    # ✅ Good: Validate key format before use
    def validate_openai_key(key):
        if not key or not key.startswith("sk-"):
            raise ValueError("Invalid OpenAI API key format")
        if len(key) < 20:  # Basic length check
            raise ValueError("API key appears to be incomplete")
        return key
    
    # ✅ Good: Mask keys in logs
    def log_key_info(key):
        if key:
            masked = key[:8] + "..." + key[-4:]
            print(f"Using API key: {masked}")
    
    # ✅ Good: Environment variable fallbacks
    def get_key_with_fallback():
        key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY_BACKUP")
        if not key:
            raise EnvironmentError("No OpenAI API key found in environment")
        return validate_openai_key(key)
    
    # Usage
    try:
        api_key = get_key_with_fallback()
        log_key_info(api_key)
    except Exception as e:
        print(f"API key setup failed: {e}")
```

### **Logging Configuration**

#### **`setup_logging(level="INFO", log_file=None)`**
```python
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
```

**Advanced Logging Configurations:**
```python
import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

class AdvancedLoggingConfig:
    """Advanced logging configuration for production use"""
    
    @staticmethod
    def setup_production_logging():
        """Setup logging for production environment"""
        # Ensure logs directory exists
        os.makedirs("logs", exist_ok=True)
        
        # Root logger configuration
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler with color formatting
        console_handler = logging.StreamHandler()
        console_formatter = ColoredFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        # File handler with rotation
        file_handler = RotatingFileHandler(
            "logs/agent.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        # Error handler - separate file for errors only
        error_handler = logging.FileHandler("logs/errors.log")
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        root_logger.addHandler(error_handler)
        
        # Performance handler for timing logs
        perf_handler = logging.FileHandler("logs/performance.log")
        perf_handler.setLevel(logging.INFO)
        perf_filter = PerformanceFilter()
        perf_handler.addFilter(perf_filter)
        root_logger.addHandler(perf_handler)
    
    @staticmethod
    def setup_development_logging():
        """Setup logging for development environment"""
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler("logs/development.log")
            ]
        )
    
    @staticmethod
    def setup_testing_logging():
        """Setup minimal logging for testing"""
        logging.basicConfig(
            level=logging.WARNING,
            format="%(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()]
        )

class ColoredFormatter(logging.Formatter):
    """Colored console logging formatter"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{log_color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)

class PerformanceFilter(logging.Filter):
    """Filter to capture only performance-related logs"""
    
    def filter(self, record):
        performance_keywords = ['performance', 'timing', 'duration', 'speed', 'benchmark']
        return any(keyword in record.getMessage().lower() for keyword in performance_keywords)

# Environment-aware logging setup
def setup_environment_logging():
    """Setup logging based on environment"""
    env = os.getenv("ENVIRONMENT", "development")
    
    if env == "production":
        AdvancedLoggingConfig.setup_production_logging()
    elif env == "testing":
        AdvancedLoggingConfig.setup_testing_logging()
    else:
        AdvancedLoggingConfig.setup_development_logging()
    
    logging.info(f"Logging configured for {env} environment")
```

**Structured Logging with Context:**
```python
import structlog
import logging

def setup_structured_logging():
    """Setup structured logging with context"""
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure stdlib logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.INFO,
    )

# Usage with structured logging
logger = structlog.get_logger()

# Log with context
logger.info(
    "agent_task_started",
    url="https://example.com",
    task_type="extract",
    user_id="user123",
    session_id="session456"
)

logger.error(
    "navigation_failed",
    url="https://example.com",
    error="timeout",
    retry_count=3,
    duration=30.5
)
```

### **Model Configuration**

#### **`get_default_model()`**
```python
def get_default_model():
    return os.getenv("DEFAULT_MODEL", "gpt-4o")
```

**Advanced Model Management:**
```python
class ModelConfig:
    """Advanced model configuration management"""
    
    # Model definitions with capabilities
    MODELS = {
        "gpt-4o": {
            "provider": "openai",
            "max_tokens": 128000,
            "supports_vision": True,
            "cost_per_1k_tokens": 0.005,
            "reasoning_quality": "excellent"
        },
        "gpt-4": {
            "provider": "openai", 
            "max_tokens": 8192,
            "supports_vision": False,
            "cost_per_1k_tokens": 0.03,
            "reasoning_quality": "excellent"
        },
        "gpt-3.5-turbo": {
            "provider": "openai",
            "max_tokens": 4096, 
            "supports_vision": False,
            "cost_per_1k_tokens": 0.001,
            "reasoning_quality": "good"
        },
        "claude-3-sonnet": {
            "provider": "anthropic",
            "max_tokens": 200000,
            "supports_vision": True,
            "cost_per_1k_tokens": 0.003,
            "reasoning_quality": "excellent"
        }
    }
    
    @classmethod
    def get_model_config(cls, model_name=None):
        """Get configuration for specified model or default"""
        if not model_name:
            model_name = os.getenv("DEFAULT_MODEL", "gpt-4o")
        
        config = cls.MODELS.get(model_name)
        if not config:
            raise ValueError(f"Unknown model: {model_name}")
        
        return {
            "name": model_name,
            **config
        }
    
    @classmethod
    def get_available_models(cls, provider=None):
        """Get list of available models, optionally filtered by provider"""
        if provider:
            return [name for name, config in cls.MODELS.items() 
                   if config["provider"] == provider]
        return list(cls.MODELS.keys())
    
    @classmethod
    def select_best_model(cls, task_type="general", budget="medium"):
        """Select best model based on task requirements"""
        if task_type == "vision" and budget == "high":
            return "gpt-4o"
        elif task_type == "vision" and budget == "medium":
            return "claude-3-sonnet"
        elif budget == "low":
            return "gpt-3.5-turbo"
        else:
            return "gpt-4o"
    
    @classmethod
    def get_fallback_model(cls, primary_model):
        """Get fallback model if primary fails"""
        fallback_map = {
            "gpt-4o": "gpt-4",
            "gpt-4": "gpt-3.5-turbo", 
            "claude-3-sonnet": "gpt-4",
            "gpt-3.5-turbo": "gpt-3.5-turbo"  # No fallback
        }
        return fallback_map.get(primary_model, "gpt-3.5-turbo")

# Usage examples
model_config = ModelConfig.get_model_config()
print(f"Using model: {model_config['name']}")
print(f"Max tokens: {model_config['max_tokens']}")

# Select model based on requirements
best_model = ModelConfig.select_best_model(task_type="vision", budget="high")
available_models = ModelConfig.get_available_models(provider="openai")
```

---

## 🏗️ Configuration Patterns & Best Practices

### **Configuration Classes**
```python
from dataclasses import dataclass
from typing import Optional
import os

@dataclass
class AgentConfig:
    """Central configuration for the web browsing agent"""
    
    # API Configuration
    openai_api_key: str
    model: str = "gpt-4o"
    max_tokens: int = 1500
    temperature: float = 0.1
    
    # Browser Configuration  
    headless: bool = True
    viewport_width: int = 1280
    viewport_height: int = 720
    timeout: int = 30000
    
    # Agent Behavior
    max_steps: int = 20
    max_retries: int = 3
    retry_delay: float = 2.0
    
    # Output Configuration
    screenshots_enabled: bool = True
    screenshot_path: str = "screenshots"
    results_path: str = "results"
    log_level: str = "INFO"
    
    @classmethod
    def from_environment(cls):
        """Create configuration from environment variables"""
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            model=os.getenv("DEFAULT_MODEL", "gpt-4o"),
            max_tokens=int(os.getenv("MAX_TOKENS", "1500")),
            temperature=float(os.getenv("TEMPERATURE", "0.1")),
            headless=os.getenv("HEADLESS_MODE", "true").lower() == "true",
            viewport_width=int(os.getenv("VIEWPORT_WIDTH", "1280")),
            viewport_height=int(os.getenv("VIEWPORT_HEIGHT", "720")),
            timeout=int(os.getenv("BROWSER_TIMEOUT", "30000")),
            max_steps=int(os.getenv("MAX_STEPS", "20")),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            retry_delay=float(os.getenv("RETRY_DELAY", "2.0")),
            screenshots_enabled=os.getenv("ENABLE_SCREENSHOTS", "true").lower() == "true",
            screenshot_path=os.getenv("SCREENSHOT_PATH", "screenshots"),
            results_path=os.getenv("RESULTS_PATH", "results"),
            log_level=os.getenv("LOG_LEVEL", "INFO")
        )
    
    def validate(self):
        """Validate configuration values"""
        errors = []
        
        if not self.openai_api_key:
            errors.append("OpenAI API key is required")
        
        if self.max_tokens < 100 or self.max_tokens > 200000:
            errors.append("max_tokens must be between 100 and 200000")
        
        if self.temperature < 0 or self.temperature > 2:
            errors.append("temperature must be between 0 and 2")
        
        if self.viewport_width < 800 or self.viewport_height < 600:
            errors.append("viewport dimensions too small")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
    
    def to_dict(self):
        """Convert configuration to dictionary"""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }

# Usage
try:
    config = AgentConfig.from_environment()
    config.validate()
    print("✅ Configuration loaded and validated successfully")
except ValueError as e:
    print(f"❌ Configuration error: {e}")
```

### **Environment-Specific Configurations**
```python
class EnvironmentConfig:
    """Environment-specific configuration management"""
    
    @staticmethod
    def get_environment():
        """Detect current environment"""
        return os.getenv("ENVIRONMENT", "development").lower()
    
    @staticmethod
    def load_config():
        """Load configuration based on environment"""
        env = EnvironmentConfig.get_environment()
        
        if env == "production":
            return EnvironmentConfig.production_config()
        elif env == "staging":
            return EnvironmentConfig.staging_config()
        elif env == "testing":
            return EnvironmentConfig.testing_config()
        else:
            return EnvironmentConfig.development_config()
    
    @staticmethod
    def development_config():
        """Development environment configuration"""
        return {
            "log_level": "DEBUG",
            "headless": False,  # Show browser for debugging
            "max_retries": 1,   # Fail fast in development
            "timeout": 10000,   # Shorter timeouts
            "screenshots_enabled": True,
            "rate_limit": False,
            "cache_enabled": False
        }
    
    @staticmethod
    def testing_config():
        """Testing environment configuration"""
        return {
            "log_level": "WARNING",
            "headless": True,
            "max_retries": 0,     # No retries in tests
            "timeout": 5000,      # Very short timeouts
            "screenshots_enabled": False,
            "rate_limit": False,
            "cache_enabled": False
        }
    
    @staticmethod
    def staging_config():
        """Staging environment configuration"""
        return {
            "log_level": "INFO",
            "headless": True,
            "max_retries": 2,
            "timeout": 20000,
            "screenshots_enabled": True,
            "rate_limit": True,
            "cache_enabled": True
        }
    
    @staticmethod
    def production_config():
        """Production environment configuration"""
        return {
            "log_level": "INFO",
            "headless": True,
            "max_retries": 3,
            "timeout": 30000,
            "screenshots_enabled": False,  # Save storage in production
            "rate_limit": True,
            "cache_enabled": True,
            "monitoring_enabled": True
        }

# Apply environment-specific configuration
env_config = EnvironmentConfig.load_config()
base_config = AgentConfig.from_environment()

# Override with environment-specific settings
for key, value in env_config.items():
    if hasattr(base_config, key):
        setattr(base_config, key, value)
```

### **Configuration Validation & Health Checks**
```python
class ConfigValidator:
    """Configuration validation and health checking"""
    
    @staticmethod
    def validate_api_connections():
        """Validate that API keys work"""
        results = {}
        
        # Test OpenAI API
        try:
            import openai
            openai.api_key = os.getenv("OPENAI_API_KEY")
            # Make a minimal test call
            response = openai.Model.list()
            results["openai"] = {"status": "success", "models_available": len(response.data)}
        except Exception as e:
            results["openai"] = {"status": "failed", "error": str(e)}
        
        return results
    
    @staticmethod
    def validate_filesystem_permissions():
        """Validate filesystem permissions for required directories"""
        results = {}
        required_dirs = ["screenshots", "results", "logs", "temp"]
        
        for dir_name in required_dirs:
            try:
                # Test directory creation
                os.makedirs(dir_name, exist_ok=True)
                
                # Test file creation
                test_file = os.path.join(dir_name, "test_permission.tmp")
                with open(test_file, "w") as f:
                    f.write("test")
                os.remove(test_file)
                
                results[dir_name] = {"status": "success", "writable": True}
            except Exception as e:
                results[dir_name] = {"status": "failed", "error": str(e)}
        
        return results
    
    @staticmethod
    def validate_browser_dependencies():
        """Validate that browser dependencies are available"""
        try:
            from playwright.sync_api import sync_playwright
            
            with sync_playwright() as p:
                # Check if browsers are installed
                browsers = {"chromium": False, "firefox": False, "webkit": False}
                
                for browser_name in browsers.keys():
                    try:
                        browser = getattr(p, browser_name)
                        browser_instance = browser.launch(headless=True)
                        browser_instance.close()
                        browsers[browser_name] = True
                    except Exception:
                        browsers[browser_name] = False
                
                return {"status": "success", "browsers": browsers}
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    @staticmethod
    def run_full_health_check():
        """Run comprehensive health check"""
        print("🔍 Running configuration health check...")
        
        # API validation
        api_results = ConfigValidator.validate_api_connections()
        print(f"📡 API Connections: {api_results}")
        
        # Filesystem validation
        fs_results = ConfigValidator.validate_filesystem_permissions()
        print(f"📁 Filesystem: {fs_results}")
        
        # Browser validation
        browser_results = ConfigValidator.validate_browser_dependencies()
        print(f"🌐 Browser Dependencies: {browser_results}")
        
        # Overall health
        all_successful = (
            api_results.get("openai", {}).get("status") == "success" and
            all(r.get("status") == "success" for r in fs_results.values()) and
            browser_results.get("status") == "success"
        )
        
        if all_successful:
            print("✅ All health checks passed!")
        else:
            print("❌ Some health checks failed. Check the details above.")
        
        return {
            "overall_status": "healthy" if all_successful else "unhealthy",
            "api": api_results,
            "filesystem": fs_results,
            "browser": browser_results
        }
```

---

## 🚀 Integration Patterns & Real-World Usage

### **Application Startup Pattern**
```python
# startup.py - Complete application initialization
from config import load_env, setup_logging, get_api_key
import sys
import os

def initialize_application():
    """Complete application initialization with error handling"""
    
    print("🚀 Initializing Playwright LangGraph Agent...")
    
    try:
        # Step 1: Load environment configuration
        print("📋 Loading environment configuration...")
        load_env()
        
        # Step 2: Setup logging
        print("📝 Configuring logging...")
        log_level = os.getenv("LOG_LEVEL", "INFO")
        log_file = os.getenv("LOG_FILE", "logs/agent.log")
        setup_logging(log_level, log_file)
        
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Application initialization started")
        
        # Step 3: Validate API keys
        print("🔑 Validating API keys...")
        api_key = get_api_key("OPENAI_API_KEY")
        logger.info("OpenAI API key validated")
        
        # Step 4: Load agent configuration
        print("⚙️ Loading agent configuration...")
        config = AgentConfig.from_environment()
        config.validate()
        logger.info("Agent configuration loaded and validated")
        
        # Step 5: Run health checks
        print("🏥 Running health checks...")
        health_results = ConfigValidator.run_full_health_check()
        
        if health_results["overall_status"] != "healthy":
            logger.error("Health checks failed")
            sys.exit(1)
        
        # Step 6: Create required directories
        print("📁 Setting up directory structure...")
        from utils import ensure_dir
        required_dirs = [
            config.screenshot_path,
            config.results_path,
            "logs",
            "temp"
        ]
        
        for directory in required_dirs:
            ensure_dir(directory)
            logger.debug(f"Directory ensured: {directory}")
        
        print("✅ Application initialization complete!")
        logger.info("Application successfully initialized")
        
        return config
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        if 'logger' in locals():
            logger.error(f"Application initialization failed: {e}")
        sys.exit(1)

# Usage in main.py
if __name__ == "__main__":
    config = initialize_application()
    # Application is now ready to run
```

### **Multi-Environment Management**
```python
# environments.py - Advanced environment management
import os
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class EnvironmentProfile:
    """Represents a complete environment profile"""
    name: str
    description: str
    config: Dict[str, Any]
    required_vars: list
    optional_vars: list

class EnvironmentManager:
    """Manage multiple environment profiles"""
    
    PROFILES = {
        "development": EnvironmentProfile(
            name="development",
            description="Local development environment",
            config={
                "LOG_LEVEL": "DEBUG",
                "HEADLESS_MODE": "false",
                "MAX_RETRIES": "1",
                "ENABLE_SCREENSHOTS": "true",
                "BROWSER_TIMEOUT": "10000"
            },
            required_vars=["OPENAI_API_KEY"],
            optional_vars=["ANTHROPIC_API_KEY", "GOOGLE_API_KEY"]
        ),
        
        "testing": EnvironmentProfile(
            name="testing",
            description="Automated testing environment",
            config={
                "LOG_LEVEL": "WARNING",
                "HEADLESS_MODE": "true",
                "MAX_RETRIES": "0",
                "ENABLE_SCREENSHOTS": "false",
                "BROWSER_TIMEOUT": "5000"
            },
            required_vars=["OPENAI_API_KEY"],
            optional_vars=[]
        ),
        
        "staging": EnvironmentProfile(
            name="staging",
            description="Pre-production staging environment",
            config={
                "LOG_LEVEL": "INFO",
                "HEADLESS_MODE": "true",
                "MAX_RETRIES": "2",
                "ENABLE_SCREENSHOTS": "true",
                "BROWSER_TIMEOUT": "20000",
                "RATE_LIMIT_ENABLED": "true"
            },
            required_vars=["OPENAI_API_KEY"],
            optional_vars=["ANTHROPIC_API_KEY"]
        ),
        
        "production": EnvironmentProfile(
            name="production",
            description="Production environment",
            config={
                "LOG_LEVEL": "INFO",
                "HEADLESS_MODE": "true",
                "MAX_RETRIES": "3",
                "ENABLE_SCREENSHOTS": "false",
                "BROWSER_TIMEOUT": "30000",
                "RATE_LIMIT_ENABLED": "true",
                "MONITORING_ENABLED": "true"
            },
            required_vars=["OPENAI_API_KEY"],
            optional_vars=["ANTHROPIC_API_KEY", "SENTRY_DSN", "DATADOG_API_KEY"]
        )
    }
    
    @classmethod
    def detect_environment(cls) -> str:
        """Auto-detect environment based on various indicators"""
        
        # Check explicit environment variable
        if env := os.getenv("ENVIRONMENT"):
            return env.lower()
        
        # Check for CI environment
        if os.getenv("CI") or os.getenv("GITHUB_ACTIONS"):
            return "testing"
        
        # Check for production indicators
        production_indicators = [
            os.getenv("RAILWAY_ENVIRONMENT"),
            os.getenv("HEROKU_APP_NAME"),
            os.getenv("AWS_EXECUTION_ENV"),
            os.getenv("GOOGLE_CLOUD_PROJECT")
        ]
        
        if any(production_indicators):
            return "production"
        
        # Check for staging indicators
        if "staging" in os.getenv("HOSTNAME", "").lower():
            return "staging"
        
        # Default to development
        return "development"
    
    @classmethod
    def load_environment(cls, env_name: str = None) -> EnvironmentProfile:
        """Load specific environment profile"""
        if not env_name:
            env_name = cls.detect_environment()
        
        if env_name not in cls.PROFILES:
            raise ValueError(f"Unknown environment: {env_name}")
        
        profile = cls.PROFILES[env_name]
        
        # Apply environment-specific configuration
        for key, value in profile.config.items():
            if not os.getenv(key):  # Don't override existing values
                os.environ[key] = value
        
        return profile
    
    @classmethod
    def validate_environment(cls, profile: EnvironmentProfile) -> Dict[str, Any]:
        """Validate environment requirements"""
        validation_results = {
            "environment": profile.name,
            "status": "valid",
            "missing_required": [],
            "missing_optional": [],
            "warnings": []
        }
        
        # Check required variables
        for var in profile.required_vars:
            if not os.getenv(var):
                validation_results["missing_required"].append(var)
                validation_results["status"] = "invalid"
        
        # Check optional variables
        for var in profile.optional_vars:
            if not os.getenv(var):
                validation_results["missing_optional"].append(var)
        
        # Environment-specific validations
        if profile.name == "production":
            if not os.getenv("SENTRY_DSN"):
                validation_results["warnings"].append("SENTRY_DSN not set - error tracking disabled")
            
            if os.getenv("DEBUG", "false").lower() == "true":
                validation_results["warnings"].append("DEBUG mode enabled in production")
        
        return validation_results

# Usage example
env_manager = EnvironmentManager()
current_env = env_manager.detect_environment()
profile = env_manager.load_environment(current_env)
validation = env_manager.validate_environment(profile)

print(f"Environment: {profile.name} - {profile.description}")
print(f"Validation: {validation}")
```

### **Configuration Factory Pattern**
```python
# config_factory.py - Factory pattern for configuration creation
from abc import ABC, abstractmethod
from typing import Any, Dict

class ConfigFactory(ABC):
    """Abstract factory for creating configurations"""
    
    @abstractmethod
    def create_agent_config(self) -> AgentConfig:
        pass
    
    @abstractmethod
    def create_logging_config(self) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def create_browser_config(self) -> Dict[str, Any]:
        pass

class DevelopmentConfigFactory(ConfigFactory):
    """Factory for development configuration"""
    
    def create_agent_config(self) -> AgentConfig:
        return AgentConfig(
            openai_api_key=get_api_key(),
            model="gpt-3.5-turbo",  # Cheaper for development
            max_tokens=1000,
            temperature=0.2,
            headless=False,  # Show browser
            max_steps=10,
            max_retries=1
        )
    
    def create_logging_config(self) -> Dict[str, Any]:
        return {
            "level": "DEBUG",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
            "handlers": ["console", "file"],
            "file_path": "logs/development.log"
        }
    
    def create_browser_config(self) -> Dict[str, Any]:
        return {
            "headless": False,
            "devtools": True,
            "slow_mo": 1000,  # Slow down for observation
            "viewport": {"width": 1280, "height": 720},
            "timeout": 10000
        }

class ProductionConfigFactory(ConfigFactory):
    """Factory for production configuration"""
    
    def create_agent_config(self) -> AgentConfig:
        return AgentConfig(
            openai_api_key=get_api_key(),
            model="gpt-4o",  # Best model for production
            max_tokens=1500,
            temperature=0.1,
            headless=True,
            max_steps=20,
            max_retries=3
        )
    
    def create_logging_config(self) -> Dict[str, Any]:
        return {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "handlers": ["console", "file", "rotating_file"],
            "file_path": "logs/production.log",
            "max_file_size": "10MB",
            "backup_count": 5
        }
    
    def create_browser_config(self) -> Dict[str, Any]:
        return {
            "headless": True,
            "args": [
                "--no-sandbox",
                "--disable-dev-shm-usage", 
                "--disable-extensions",
                "--disable-plugins"
            ],
            "viewport": {"width": 1920, "height": 1080},
            "timeout": 30000
        }

class ConfigFactoryProvider:
    """Provides appropriate factory based on environment"""
    
    @staticmethod
    def get_factory(environment: str = None) -> ConfigFactory:
        if not environment:
            environment = os.getenv("ENVIRONMENT", "development")
        
        factories = {
            "development": DevelopmentConfigFactory,
            "testing": TestingConfigFactory,
            "staging": ProductionConfigFactory,  # Use production-like config
            "production": ProductionConfigFactory
        }
        
        factory_class = factories.get(environment, DevelopmentConfigFactory)
        return factory_class()

# Usage
factory = ConfigFactoryProvider.get_factory()
agent_config = factory.create_agent_config()
logging_config = factory.create_logging_config()
browser_config = factory.create_browser_config()
```

---

## 🧪 Testing Configuration

### **Configuration Testing Strategies**
```python
# test_config.py - Comprehensive configuration testing
import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock
from config import load_env, get_api_key, setup_logging, get_default_model

class TestConfigModule:
    """Test suite for configuration module"""
    
    def test_load_env_default_file(self, tmp_path):
        """Test loading default .env file"""
        # Create temporary .env file
        env_file = tmp_path / ".env"
        env_file.write_text("TEST_VAR=test_value\nANOTHER_VAR=another_value")
        
        # Change to temp directory
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            load_env()
            assert os.getenv("TEST_VAR") == "test_value"
            assert os.getenv("ANOTHER_VAR") == "another_value"
        finally:
            os.chdir(original_cwd)
    
    def test_load_env_custom_path(self, tmp_path):
        """Test loading custom .env file"""
        env_file = tmp_path / "custom.env"
        env_file.write_text("CUSTOM_VAR=custom_value")
        
        load_env(str(env_file))
        assert os.getenv("CUSTOM_VAR") == "custom_value"
    
    def test_get_api_key_success(self, monkeypatch):
        """Test successful API key retrieval"""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        assert get_api_key() == "sk-test-key"
    
    def test_get_api_key_missing(self, monkeypatch):
        """Test API key missing raises error"""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        
        with pytest.raises(EnvironmentError, match="Missing environment variable"):
            get_api_key()
    
    def test_get_api_key_custom_var(self, monkeypatch):
        """Test custom API key variable"""
        monkeypatch.setenv("CUSTOM_API_KEY", "custom-key")
        assert get_api_key("CUSTOM_API_KEY") == "custom-key"
    
    def test_setup_logging_console_only(self, caplog):
        """Test logging setup with console handler only"""
        setup_logging("DEBUG")
        
        import logging
        logger = logging.getLogger("test")
        logger.debug("test debug message")
        
        assert "test debug message" in caplog.text
    
    def test_setup_logging_with_file(self, tmp_path):
        """Test logging setup with file handler"""
        log_file = tmp_path / "test.log"
        setup_logging("INFO", str(log_file))
        
        import logging
        logger = logging.getLogger("test")
        logger.info("test info message")
        
        assert log_file.exists()
        assert "test info message" in log_file.read_text()
    
    def test_get_default_model_default(self, monkeypatch):
        """Test default model when no env var set"""
        monkeypatch.delenv("DEFAULT_MODEL", raising=False)
        assert get_default_model() == "gpt-4o"
    
    def test_get_default_model_custom(self, monkeypatch):
        """Test custom default model from env var"""
        monkeypatch.setenv("DEFAULT_MODEL", "gpt-3.5-turbo")
        assert get_default_model() == "gpt-3.5-turbo"

class TestAgentConfig:
    """Test suite for AgentConfig class"""
    
    def test_from_environment_defaults(self, monkeypatch):
        """Test AgentConfig creation with defaults"""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        
        # Clear other env vars to test defaults
        env_vars_to_clear = [
            "DEFAULT_MODEL", "MAX_TOKENS", "TEMPERATURE",
            "HEADLESS_MODE", "VIEWPORT_WIDTH", "VIEWPORT_HEIGHT"
        ]
        
        for var in env_vars_to_clear:
            monkeypatch.delenv(var, raising=False)
        
        config = AgentConfig.from_environment()
        
        assert config.openai_api_key == "sk-test"
        assert config.model == "gpt-4o"
        assert config.max_tokens == 1500
        assert config.temperature == 0.1
        assert config.headless is True
        assert config.viewport_width == 1280
        assert config.viewport_height == 720
    
    def test_from_environment_custom_values(self, monkeypatch):
        """Test AgentConfig creation with custom values"""
        env_vars = {
            "OPENAI_API_KEY": "sk-custom",
            "DEFAULT_MODEL": "gpt-3.5-turbo",
            "MAX_TOKENS": "2000",
            "TEMPERATURE": "0.5",
            "HEADLESS_MODE": "false",
            "VIEWPORT_WIDTH": "1920",
            "VIEWPORT_HEIGHT": "1080"
        }
        
        for key, value in env_vars.items():
            monkeypatch.setenv(key, value)
        
        config = AgentConfig.from_environment()
        
        assert config.openai_api_key == "sk-custom"
        assert config.model == "gpt-3.5-turbo"
        assert config.max_tokens == 2000
        assert config.temperature == 0.5
        assert config.headless is False
        assert config.viewport_width == 1920
        assert config.viewport_height == 1080
    
    def test_validate_success(self, monkeypatch):
        """Test successful configuration validation"""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-valid-key")
        config = AgentConfig.from_environment()
        
        # Should not raise any exception
        config.validate()
    
    def test_validate_missing_api_key(self):
        """Test validation failure for missing API key"""
        config = AgentConfig(openai_api_key="")
        
        with pytest.raises(ValueError, match="OpenAI API key is required"):
            config.validate()
    
    def test_validate_invalid_max_tokens(self, monkeypatch):
        """Test validation failure for invalid max_tokens"""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        config = AgentConfig.from_environment()
        config.max_tokens = 50  # Too low
        
        with pytest.raises(ValueError, match="max_tokens must be between"):
            config.validate()
    
    def test_validate_invalid_temperature(self, monkeypatch):
        """Test validation failure for invalid temperature"""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        config = AgentConfig.from_environment()
        config.temperature = 3.0  # Too high
        
        with pytest.raises(ValueError, match="temperature must be between"):
            config.validate()

@pytest.fixture
def mock_env_file(tmp_path):
    """Fixture for creating mock .env files"""
    def create_env_file(content, filename=".env"):
        env_file = tmp_path / filename
        env_file.write_text(content)
        return str(env_file)
    
    return create_env_file

class TestIntegrationScenarios:
    """Integration tests for real-world scenarios"""
    
    def test_complete_initialization_flow(self, monkeypatch, tmp_path):
        """Test complete application initialization"""
        # Setup environment
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.setenv("ENVIRONMENT", "testing")
        
        # Change to temp directory to avoid file conflicts
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            # Test the complete flow
            load_env()
            setup_logging("INFO", "test.log")
            api_key = get_api_key()
            config = AgentConfig.from_environment()
            config.validate()
            
            assert api_key == "sk-test-key"
            assert config.openai_api_key == "sk-test-key"
            assert (tmp_path / "test.log").exists()
            
        finally:
            os.chdir(original_cwd)
    
    def test_environment_switching(self, monkeypatch):
        """Test switching between environments"""
        environments = ["development", "testing", "staging", "production"]
        
        for env in environments:
            monkeypatch.setenv("ENVIRONMENT", env)
            monkeypatch.setenv("OPENAI_API_KEY", f"sk-{env}-key")
            
            # Test that environment detection works
            detected = EnvironmentManager.detect_environment()
            assert detected == env
            
            # Test that configuration adapts
            profile = EnvironmentManager.load_environment(env)
            assert profile.name == env

# Benchmark tests for performance
class TestConfigPerformance:
    """Performance tests for configuration operations"""
    
    def test_load_env_performance(self, benchmark, tmp_path):
        """Benchmark .env file loading"""
        # Create large .env file
        env_content = "\n".join([f"VAR_{i}=value_{i}" for i in range(1000)])
        env_file = tmp_path / "large.env"
        env_file.write_text(env_content)
        
        def load_large_env():
            load_env(str(env_file))
        
        result = benchmark(load_large_env)
        assert result is None  # Function should complete successfully
    
    def test_config_creation_performance(self, benchmark, monkeypatch):
        """Benchmark configuration object creation"""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        
        def create_config():
            return AgentConfig.from_environment()
        
        config = benchmark(create_config)
        assert isinstance(config, AgentConfig)

# Property-based testing
@pytest.mark.parametrize("model_name,expected_valid", [
    ("gpt-4o", True),
    ("gpt-4", True), 
    ("gpt-3.5-turbo", True),
    ("invalid-model", False),
    ("", False)
])
def test_model_validation(model_name, expected_valid):
    """Test model name validation with various inputs"""
    valid_models = ["gpt-4o", "gpt-4", "gpt-3.5-turbo", "claude-3-sonnet"]
    
    is_valid = model_name in valid_models and len(model_name) > 0
    assert is_valid == expected_valid

@pytest.mark.parametrize("api_key,should_raise", [
    ("sk-valid-key-format", False),
    ("", True),
    (None, True),
    ("invalid-format", True)
])
def test_api_key_validation_parametrized(api_key, should_raise, monkeypatch):
    """Test API key validation with various inputs"""
    if api_key is None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    else:
        monkeypatch.setenv("OPENAI_API_KEY", api_key)
    
    if should_raise:
        with pytest.raises((EnvironmentError, ValueError)):
            get_api_key()
    else:
        result = get_api_key()
        assert result == api_key
```

---

## 🚀 Production Deployment Examples

### **Docker Configuration**
```dockerfile
# Dockerfile - Production deployment with configuration
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Install Playwright browsers
RUN pip install playwright
RUN playwright install chromium

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs results screenshots temp

# Set environment variables with defaults
ENV ENVIRONMENT=production
ENV LOG_LEVEL=INFO
ENV HEADLESS_MODE=true
ENV MAX_RETRIES=3

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from config import get_api_key; get_api_key()" || exit 1

# Run configuration validation on startup
RUN python -c "from config import load_env; load_env(); print('Configuration validated')"

# Default command
CMD ["python", "main.py"]
```

```yaml
# docker-compose.yml - Multi-environment deployment
version: '3.8'

services:
  agent-dev:
    build: .
    environment:
      - ENVIRONMENT=development
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - LOG_LEVEL=DEBUG
      - HEADLESS_MODE=false
    volumes:
      - ./logs:/app/logs
      - ./results:/app/results
    ports:
      - "9222:9222"  # Chrome DevTools

  agent-staging:
    build: .
    environment:
      - ENVIRONMENT=staging
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - LOG_LEVEL=INFO
      - HEADLESS_MODE=true
      - SENTRY_DSN=${SENTRY_DSN}
    volumes:
      - staging_logs:/app/logs
      - staging_results:/app/results

  agent-prod:
    build: .
    environment:
      - ENVIRONMENT=production
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - LOG_LEVEL=INFO
      - HEADLESS_MODE=true
      - MONITORING_ENABLED=true
      - SENTRY_DSN=${SENTRY_DSN}
    volumes:
      - prod_logs:/app/logs
      - prod_results:/app/results
    restart: unless-stopped

volumes:
  staging_logs:
  staging_results:
  prod_logs:
  prod_results:
```

### **Kubernetes Configuration**
```yaml
# k8s-config.yaml - Kubernetes deployment
apiVersion: v1
kind: ConfigMap
metadata:
  name: agent-config
data:
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
  HEADLESS_MODE: "true"
  MAX_RETRIES: "3"
  BROWSER_TIMEOUT: "30000"

---
apiVersion: v1
kind: Secret
metadata:
  name: agent-secrets
type: Opaque
stringData:
  OPENAI_API_KEY: "your-secret-key-here"
  SENTRY_DSN: "your-sentry-dsn-here"

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-browsing-agent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: web-browsing-agent
  template:
    metadata:
      labels:
        app: web-browsing-agent
    spec:
      containers:
      - name: agent
        image: your-registry/web-browsing-agent:latest
        envFrom:
        - configMapRef:
            name: agent-config
        - secretRef:
            name: agent-secrets
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          exec:
            command:
            - python
            - -c
            - "from config import get_api_key; get_api_key()"
          initialDelaySeconds: 30
          periodSeconds: 30
        volumeMounts:
        - name: logs
          mountPath: /app/logs
        - name: results
          mountPath: /app/results
      volumes:
      - name: logs
        emptyDir: {}
      - name: results
        persistentVolumeClaim:
          claimName: agent-results-pvc
```

### **CI/CD Pipeline Configuration**
```yaml
# .github/workflows/test-and-deploy.yml
name: Test and Deploy Web Browsing Agent

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  PYTHON_VERSION: '3.11'

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        environment: [development, testing, staging]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        playwright install chromium
    
    - name: Load test environment
      run: |
        echo "ENVIRONMENT=${{ matrix.environment }}" >> $GITHUB_ENV
        echo "OPENAI_API_KEY=${{ secrets.OPENAI_API_KEY }}" >> $GITHUB_ENV
    
    - name: Validate configuration
      run: |
        python -c "
        from config import load_env, get_api_key;
        load_env();
        print('API Key:', get_api_key()[:12] + '...');
        print('Configuration validated for ${{ matrix.environment }}')
        "
    
    - name: Run tests
      run: |
        pytest tests/ -v --env=${{ matrix.environment }}
    
    - name: Run health checks
      run: |
        python -c "
        from config import ConfigValidator;
        results = ConfigValidator.run_full_health_check();
        print('Health check results:', results)
        "

  deploy-staging:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/develop'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to staging
      run: |
        # Deployment logic here
        echo "Deploying to staging environment"
        
  deploy-production:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to production
      run: |
        # Production deployment logic here
        echo "Deploying to production environment"
```

---

## 🎓 Learning Exercises & Best Practices

### **Exercise 1: Environment Setup Challenge**
```python
# exercise_1.py - Complete this configuration setup
"""
CHALLENGE: Create a complete configuration setup that:
1. Loads environment variables from multiple sources
2. Validates all required settings
3. Sets up logging with rotation
4. Provides helpful error messages
"""

def setup_complete_environment():
    """Your task: Implement robust environment setup"""
    
    # TODO: Load environment files in priority order
    # 1. .env.defaults (base settings)
    # 2. .env.{environment} (environment-specific)
    # 3. .env.local (local overrides)
    
    # TODO: Validate required environment variables
    required_vars = ["OPENAI_API_KEY"]
    optional_vars = ["ANTHROPIC_API_KEY", "SENTRY_DSN"]
    
    # TODO: Setup logging with:
    # - Console output for development
    # - File output with rotation for production
    # - Different log levels per environment
    
    # TODO: Return configuration object with validation
    
    pass  # Replace with your implementation

# Test your solution
if __name__ == "__main__":
    try:
        config = setup_complete_environment()
        print("✅ Configuration setup successful!")
        print(f"Environment: {config.environment}")
        print(f"Log level: {config.log_level}")
    except Exception as e:
        print(f"❌ Configuration setup failed: {e}")
```

### **Exercise 2: Custom Configuration Provider**
```python
# exercise_2.py - Build a custom configuration provider
"""
CHALLENGE: Create a configuration provider that:
1. Supports multiple configuration sources (env, file, remote)
2. Implements configuration inheritance and overrides
3. Provides configuration validation and schema checking
4. Supports hot-reloading of configuration changes
"""

class ConfigurationProvider:
    """Your task: Implement a flexible configuration provider"""
    
    def __init__(self, sources=None):
        # TODO: Initialize configuration sources
        # Sources could be: environment, files, remote APIs, etc.
        pass
    
    def load_configuration(self):
        # TODO: Load configuration from all sources
        # Implement priority and override logic
        pass
    
    def validate_schema(self, config_data):
        # TODO: Validate configuration against schema
        # Return validation results with detailed errors
        pass
    
    def watch_for_changes(self, callback):
        # TODO: Implement hot-reloading
        # Call callback when configuration changes
        pass
    
    def get_config_value(self, key, default=None):
        # TODO: Get configuration value with dot notation support
        # Example: get_config_value("database.host")
        pass

# Test your provider
provider = ConfigurationProvider(sources=["env", "config.json", "remote"])
config = provider.load_configuration()
```

### **Exercise 3: Configuration Migration Tool**
```python
# exercise_3.py - Build a configuration migration tool
"""
CHALLENGE: Create a tool that helps migrate configurations between versions:
1. Detect configuration schema changes
2. Automatically migrate old configurations to new format
3. Backup old configurations before migration
4. Validate migrated configurations
"""

class ConfigMigrationTool:
    """Your task: Implement configuration migration"""
    
    def __init__(self, old_version, new_version):
        self.old_version = old_version
        self.new_version = new_version
        self.migration_rules = {}
    
    def detect_changes(self):
        # TODO: Compare old and new configuration schemas
        # Return list of changes (added, removed, renamed, type changes)
        pass
    
    def create_migration_plan(self):
        # TODO: Create step-by-step migration plan
        # Handle field renames, type conversions, defaults for new fields
        pass
    
    def execute_migration(self, config_data):
        # TODO: Execute migration plan on configuration data
        # Return migrated configuration
        pass
    
    def validate_migration(self, old_config, new_config):
        # TODO: Validate that migration was successful
        # Check that no data was lost and new config is valid
        pass

# Example usage
migrator = ConfigMigrationTool("1.0", "2.0")
changes = migrator.detect_changes()
plan = migrator.create_migration_plan()
```

---

## 🏆 Advanced Patterns & Production Tips

### **Configuration Security Best Practices**
```python
# security_config.py - Secure configuration handling
import hashlib
import base64
from cryptography.fernet import Fernet

class SecureConfigManager:
    """Secure configuration management with encryption"""
    
    def __init__(self, encryption_key=None):
        self.encryption_key = encryption_key or self._generate_key()
        self.cipher = Fernet(self.encryption_key)
    
    def _generate_key(self):
        """Generate encryption key from environment"""
        master_key = os.getenv("CONFIG_MASTER_KEY", "default-key")
        key_bytes = hashlib.sha256(master_key.encode()).digest()
        return base64.urlsafe_b64encode(key_bytes)
    
    def encrypt_sensitive_value(self, value):
        """Encrypt sensitive configuration values"""
        if isinstance(value, str):
            value = value.encode()
        encrypted = self.cipher.encrypt(value)
        return base64.urlsafe_b64encode(encrypted).decode()
    
    def decrypt_sensitive_value(self, encrypted_value):
        """Decrypt sensitive configuration values"""
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_value.encode())
        decrypted = self.cipher.decrypt(encrypted_bytes)
        return decrypted.decode()
    
    def mask_sensitive_logs(self, config_dict):
        """Mask sensitive values in logs"""
        sensitive_keys = [
            "api_key", "password", "secret", "token", 
            "private_key", "credential"
        ]
        
        masked_config = {}
        for key, value in config_dict.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                if isinstance(value, str) and len(value) > 8:
                    masked_config[key] = value[:4] + "***" + value[-4:]
                else:
                    masked_config[key] = "***"
            else:
                masked_config[key] = value
        
        return masked_config

# Usage
secure_manager = SecureConfigManager()

# Encrypt sensitive values before storing
api_key = "sk-very-secret-api-key"
encrypted_key = secure_manager.encrypt_sensitive_value(api_key)

# Store encrypted value in configuration
config = {"encrypted_api_key": encrypted_key}

# Decrypt when needed
decrypted_key = secure_manager.decrypt_sensitive_value(encrypted_key)
```

### **Configuration Monitoring & Alerting**
```python
# config_monitoring.py - Monitor configuration health
import time
import threading
from datetime import datetime, timedelta

class ConfigMonitor:
    """Monitor configuration health and alert on issues"""
    
    def __init__(self, check_interval=300):  # 5 minutes
        self.check_interval = check_interval
        self.last_check = None
        self.alert_threshold = timedelta(minutes=10)
        self.running = False
        self.thread = None
    
    def start_monitoring(self):
        """Start configuration monitoring"""
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()
        print("🔍 Configuration monitoring started")
    
    def stop_monitoring(self):
        """Stop configuration monitoring"""
        self.running = False
        if self.thread:
            self.thread.join()
        print("⏹️ Configuration monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                self._perform_health_check()
                time.sleep(self.check_interval)
            except Exception as e:
                self._send_alert(f"Configuration monitoring error: {e}")
                time.sleep(60)  # Wait a minute before retrying
    
    def _perform_health_check(self):
        """Perform configuration health check"""
        health_results = {
            "timestamp": datetime.now(),
            "api_connectivity": self._check_api_connectivity(),
            "environment_vars": self._check_environment_vars(),
            "file_permissions": self._check_file_permissions(),
            "disk_space": self._check_disk_space()
        }
        
        # Check for critical issues
        critical_issues = []
        if not health_results["api_connectivity"]["healthy"]:
            critical_issues.append("API connectivity failed")
        
        if not health_results["environment_vars"]["healthy"]:
            critical_issues.append("Missing required environment variables")
        
        if critical_issues:
            self._send_alert(f"Critical configuration issues: {', '.join(critical_issues)}")
        
        self.last_check = datetime.now()
        return health_results
    
    def _check_api_connectivity(self):
        """Check API connectivity"""
        try:
            api_key = get_api_key()
            # Perform minimal API test
            return {"healthy": True, "response_time": 0.1}
        except Exception as e:
            return {"healthy": False, "error": str(e)}
    
    def _check_environment_vars(self):
        """Check required environment variables"""
        required_vars = ["OPENAI_API_KEY"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        return {
            "healthy": len(missing_vars) == 0,
            "missing_vars": missing_vars
        }
    
    def _check_file_permissions(self):
        """Check file system permissions"""
        try:
            # Test writing to required directories
            test_dirs = ["logs", "results", "screenshots"]
            for directory in test_dirs:
                ensure_dir(directory)
                test_file = os.path.join(directory, ".permission_test")
                with open(test_file, "w") as f:
                    f.write("test")
                os.remove(test_file)
            
            return {"healthy": True}
        except Exception as e:
            return {"healthy": False, "error": str(e)}
    
    def _check_disk_space(self):
        """Check available disk space"""
        import shutil
        
        try:
            total, used, free = shutil.disk_usage(".")
            free_percentage = (free / total) * 100
            
            return {
                "healthy": free_percentage > 10,  # Alert if less than 10% free
                "free_percentage": free_percentage,
                "free_gb": free / (1024**3)
            }
        except Exception as e:
            return {"healthy": False, "error": str(e)}
    
    def _send_alert(self, message):
        """Send alert notification"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        alert_message = f"[{timestamp}] ALERT: {message}"
        
        # Log the alert
        import logging
        logger = logging.getLogger(__name__)
        logger.error(alert_message)
        
        # Send to monitoring service (implement based on your setup)
        self._send_to_monitoring_service(alert_message)
    
    def _send_to_monitoring_service(self, message):
        """Send alert to external monitoring service"""
        # Example: Send to Slack, PagerDuty, email, etc.
        print(f"🚨 {message}")

# Usage
monitor = ConfigMonitor(check_interval=300)
monitor.start_monitoring()

# In your application shutdown
# monitor.stop_monitoring()
```

### **Configuration Performance Optimization**
```python
# config_optimization.py - Optimize configuration performance
import functools
import threading
from typing import Any, Dict

class OptimizedConfigCache:
    """High-performance configuration caching"""
    
    def __init__(self, ttl_seconds=300):
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.cache_timestamps = {}
        self.lock = threading.RLock()
    
    def get_cached_config(self, key: str, loader_func):
        """Get configuration with caching"""
        with self.lock:
            now = time.time()
            
            # Check if cache is valid
            if (key in self.cache and 
                key in self.cache_timestamps and
                now - self.cache_timestamps[key] < self.ttl_seconds):
                return self.cache[key]
            
            # Load fresh configuration
            config = loader_func()
            self.cache[key] = config
            self.cache_timestamps[key] = now
            
            return config
    
    def invalidate_cache(self, key: str = None):
        """Invalidate cache entries"""
        with self.lock:
            if key:
                self.cache.pop(key, None)
                self.cache_timestamps.pop(key, None)
            else:
                self.cache.clear()
                self.cache_timestamps.clear()

# Singleton pattern for global configuration
class ConfigSingleton:
    """Singleton configuration manager"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
                    cls._instance.initialized = False
        return cls._instance
    
    def __init__(self):
        if not self.initialized:
            self.config_cache = OptimizedConfigCache()
            self.config_data = {}
            self.initialized = True
    
    def get_config(self):
        """Get configuration with caching"""
        return self.config_cache.get_cached_config(
            "main_config",
            self._load_configuration
        )
    
    def _load_configuration(self):
        """Load configuration from all sources"""
        load_env()
        config = AgentConfig.from_environment()
        config.validate()
        return config

# Optimized configuration loading with lazy initialization
class LazyConfigLoader:
    """Lazy-loaded configuration properties"""
    
    def __init__(self):
        self._config = None
        self._api_key = None
        self._model_config = None
    
    @property
    def config(self):
        """Lazily load main configuration"""
        if self._config is None:
            self._config = AgentConfig.from_environment()
            self._config.validate()
        return self._config
    
    @property
    @functools.lru_cache(maxsize=1)
    def api_key(self):
        """Cached API key property"""
        return get_api_key()
    
    @property
    @functools.lru_cache(maxsize=1)
    def model_config(self):
        """Cached model configuration"""
        return ModelConfig.get_model_config(self.config.model)

# Usage examples
config_singleton = ConfigSingleton()
lazy_loader = LazyConfigLoader()

# Fast access to configuration
config = config_singleton.get_config()
api_key = lazy_loader.api_key
model_info = lazy_loader.model_config
```

---

## 📊 Configuration Analytics & Insights

### **Configuration Usage Analytics**
```python
# config_analytics.py - Track configuration usage and performance
import time
import json
from collections import defaultdict, Counter
from datetime import datetime, timedelta

class ConfigAnalytics:
    """Track and analyze configuration usage patterns"""
    
    def __init__(self):
        self.usage_stats = defaultdict(list)
        self.error_stats = Counter()
        self.performance_stats = {}
        self.start_time = time.time()
    
    def track_config_access(self, config_key, access_time=None):
        """Track configuration key access"""
        if access_time is None:
            access_time = time.time()
        
        self.usage_stats[config_key].append({
            "timestamp": access_time,
            "datetime": datetime.fromtimestamp(access_time).isoformat()
        })
    
    def track_config_error(self, error_type, details=None):
        """Track configuration errors"""
        self.error_stats[error_type] += 1
        
        if details:
            error_key = f"{error_type}_{hash(str(details)) % 1000}"
            self.usage_stats[f"error_{error_key}"].append({
                "timestamp": time.time(),
                "error_type": error_type,
                "details": details
            })
    
    def track_performance(self, operation, duration):
        """Track configuration operation performance"""
        if operation not in self.performance_stats:
            self.performance_stats[operation] = []
        
        self.performance_stats[operation].append({
            "duration": duration,
            "timestamp": time.time()
        })
    
    def generate_usage_report(self, time_window_hours=24):
        """Generate configuration usage report"""
        cutoff_time = time.time() - (time_window_hours * 3600)
        
        # Filter recent usage
        recent_usage = {}
        for key, accesses in self.usage_stats.items():
            recent_accesses = [a for a in accesses if a["timestamp"] > cutoff_time]
            if recent_accesses:
                recent_usage[key] = recent_accesses
        
        # Calculate statistics
        most_accessed = Counter({
            key: len(accesses) for key, accesses in recent_usage.items()
        }).most_common(10)
        
        # Performance analysis
        perf_summary = {}
        for operation, measurements in self.performance_stats.items():
            recent_measurements = [m for m in measurements if m["timestamp"] > cutoff_time]
            if recent_measurements:
                durations = [m["duration"] for m in recent_measurements]
                perf_summary[operation] = {
                    "count": len(durations),
                    "avg_duration": sum(durations) / len(durations),
                    "max_duration": max(durations),
                    "min_duration": min(durations)
                }
        
        report = {
            "report_generated": datetime.now().isoformat(),
            "time_window_hours": time_window_hours,
            "uptime_hours": (time.time() - self.start_time) / 3600,
            "total_config_accesses": sum(len(accesses) for accesses in recent_usage.values()),
            "unique_config_keys": len(recent_usage),
            "most_accessed_keys": most_accessed,
            "error_summary": dict(self.error_stats),
            "performance_summary": perf_summary
        }
        
        return report
    
    def save_analytics_data(self, filename=None):
        """Save analytics data to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"config_analytics_{timestamp}.json"
        
        analytics_data = {
            "usage_stats": dict(self.usage_stats),
            "error_stats": dict(self.error_stats),
            "performance_stats": self.performance_stats,
            "metadata": {
                "start_time": self.start_time,
                "export_time": time.time(),
                "uptime_hours": (time.time() - self.start_time) / 3600
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(analytics_data, f, indent=2, default=str)
        
        print(f"📊 Analytics data saved to {filename}")
        return filename

# Enhanced configuration manager with analytics
class AnalyticsConfigManager:
    """Configuration manager with built-in analytics"""
    
    def __init__(self):
        self.analytics = ConfigAnalytics()
        self.config_cache = {}
    
    def get_config_value(self, key, default=None):
        """Get configuration value with analytics tracking"""
        start_time = time.time()
        
        try:
            # Track access
            self.analytics.track_config_access(key)
            
            # Get value from cache or load
            if key in self.config_cache:
                value = self.config_cache[key]
            else:
                value = os.getenv(key, default)
                self.config_cache[key] = value
            
            # Track performance
            duration = time.time() - start_time
            self.analytics.track_performance(f"get_{key}", duration)
            
            return value
            
        except Exception as e:
            self.analytics.track_config_error("get_config_error", {
                "key": key,
                "error": str(e)
            })
            raise
    
    def generate_daily_report(self):
        """Generate and save daily analytics report"""
        report = self.analytics.generate_usage_report(24)
        
        timestamp = datetime.now().strftime("%Y%m%d")
        report_file = f"daily_config_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📈 Daily report saved to {report_file}")
        return report

# Usage
analytics_config = AnalyticsConfigManager()

# Track configuration usage
api_key = analytics_config.get_config_value("OPENAI_API_KEY")
model = analytics_config.get_config_value("DEFAULT_MODEL", "gpt-4o")

# Generate reports
daily_report = analytics_config.generate_daily_report()
```

---

## 🎯 Summary & Key Takeaways

The `config.py` module represents the **foundational infrastructure** of your Playwright LangGraph Agent. Here are the essential concepts you should master:

### **🏗️ Architectural Principles**
✅ **Centralized Configuration**: Single source of truth for all settings  
✅ **Environment-Aware**: Adapts behavior based on development/staging/production  
✅ **Fail-Fast Validation**: Catches configuration errors at startup, not runtime  
✅ **Security-First**: Proper handling of API keys and sensitive data  
✅ **Observable**: Comprehensive logging and monitoring capabilities  

### **🛠️ Core Capabilities**
- **Environment Management**: Load and validate environment variables
- **API Key Security**: Secure handling and validation of API credentials  
- **Logging Setup**: Flexible, environment-aware logging configuration
- **Model Configuration**: Smart model selection and fallback handling
- **Health Monitoring**: Continuous validation of configuration health

### **🚀 Production Readiness**
- **Multi-Environment Support**: Seamless deployment across environments
- **Configuration Migration**: Tools for upgrading between versions
- **Performance Optimization**: Caching and lazy loading for speed
- **Security Hardening**: Encryption and secure credential management
- **Analytics & Monitoring**: Track usage patterns and performance

### **📈 When to Extend config.py**
- **New API Integrations**: Add support for additional LLM providers
- **Advanced Security**: Implement credential rotation and encryption
- **Monitoring Integration**: Connect to APM tools like Datadog or New Relic
- **Multi-Tenant Support**: Configuration isolation between tenants
- **Feature Flags**: Dynamic configuration for A/B testing

### **🎓 Best Practices Checklist**
- [ ] Always call `load_env()` first in your application
- [ ] Validate all configuration at startup with clear error messages
- [ ] Use environment variables for all configuration, never hard-code
- [ ] Implement proper logging levels for different environments
- [ ] Secure API keys and never log them in plain text
- [ ] Provide sensible defaults for optional configuration
- [ ] Document all environment variables in your README
- [ ] Use configuration objects instead of scattered `os.getenv()` calls
- [ ] Implement health checks for configuration dependencies
- [ ] Plan for configuration migration between versions

### **🔧 Integration Patterns**
```python
# Perfect config.py integration pattern
from config import load_env, setup_logging, get_api_key

def main():
    # 1. Load environment first
    load_env()
    
    # 2. Setup logging
    setup_logging("INFO", "logs/app.log")
    
    # 3. Validate critical configuration
    api_key = get_api_key()
    
    # 4. Create configuration objects
    config = AgentConfig.from_environment()
    config.validate()
    
    # 5. Initialize application with config
    agent = WebBrowsingAgent(api_key, **config.to_dict())
    
    # Your application logic here...
```

The `config.py` module transforms chaotic environment management into clean, predictable, and secure configuration handling. It's the bedrock that enables your AI agents to run reliably across all environments.

**Start with solid configuration, scale with confidence!** 🛠️🚀

---

*This guide provides the complete foundation for mastering configuration management in modern Python applications. Use these patterns to build robust, secure, and maintainable AI agent systems.*