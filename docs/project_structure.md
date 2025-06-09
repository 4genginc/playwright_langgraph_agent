# 🕷️ Playwright LangGraph Agent - Complete Project Structure

## 📁 Project Directory Layout

```
playwright_langgraph_agent/
├── 📄 README.md                           # Main project documentation
├── 📄 GRADIO_UI_README.md                 # Gradio UI specific documentation
├── 🔧 .env                                # Environment variables (create this)
├── 🔧 .gitignore                          # Git ignore patterns
├── 📋 requirements.txt                    # Core Python dependencies
├── 📋 requirements_gradio.txt             # Extended dependencies for Gradio UI
├── 🚀 main.py                            # CLI entry point
├── ⚙️ config.py                          # Environment & logging configuration
├── 📊 state.py                           # BrowserState dataclass definitions
├── 🛠️ utils.py                           # General utility functions
│
├── 🌐 gradio_ui.py                       # Main Gradio web interface
├── 🚀 launch_gradio.py                   # Smart Gradio launcher script
├── 📈 gradio_extensions.py               # Advanced analytics & export features
│
├── 🤖 agent/                             # Core AI agent logic
│   └── 📄 web_browsing_agent.py          # Main WebBrowsingAgent (LangGraph)
│
├── 🌐 browser/                           # Browser automation layer
│   └── 📄 playwright_manager.py          # PlaywrightManager (async browser ops)
│
├── 🧰 toolkit/                           # Batch processing & export utilities
│   └── 📄 web_toolkit.py                 # Batch runs, CSV/JSON export
│
├── 💡 examples/                          # Example workflows & demos
│   ├── 📄 demo_tasks.py                  # Ready-to-run demo workflows
│   └── 📄 utils_examples.py              # Utility function examples
│
├── 🧪 tests/                             # Test suite
│   ├── 📄 test_config.py                 # Configuration tests
│   ├── 📄 test_playwright.py             # Browser smoke tests
│   ├── 📄 test_playwright_manager.py     # Browser manager tests
│   ├── 📄 test_web_browsing_agent.py     # Full agent pipeline tests
│   └── 📄 test_web_toolkit.py            # Export functionality tests
│
└── 📁 results/                           # Generated outputs (auto-created)
    ├── 📁 gradio_sessions/               # Single task results from UI
    │   ├── session_20240115_103000.json
    │   └── session_20240115_104500.csv
    ├── 📁 batch_runs/                    # Batch processing results
    │   └── batch_20240115_105000/
    │       ├── results.json
    │       ├── results.csv
    │       └── individual_001.json
    ├── 📁 screenshots/                   # Automatic screenshots
    │   ├── screenshot_20240115_103015.png
    │   └── screenshot_20240115_104500.png
    └── 📁 exports/                       # Manual exports & reports
        ├── analytics_report_20240115.json
        └── excel_export_20240115.xlsx
```

## 📚 Module Documentation

### 🔧 Core Configuration & Setup

| File | Purpose | Key Features |
|------|---------|--------------|
| `config.py` | Environment setup, logging config | `.env` loading, API key management, logging setup |
| `state.py` | Agent state management | BrowserState dataclass, task tracking, error handling |
| `utils.py` | General utilities | File operations, timestamps, JSON handling, logging |
| `.env` | Environment variables | API keys, model settings (user creates) |
| `requirements*.txt` | Dependencies | Core + Gradio-specific packages |

### 🤖 AI Agent Core

| File | Purpose | Key Features |
|------|---------|--------------|
| `agent/web_browsing_agent.py` | Main AI agent logic | LangGraph orchestration, LLM decision making, task execution |
| `browser/playwright_manager.py` | Browser automation | Async browser ops, element interaction, screenshot capture |
| `state.py` | State management | Task progress, navigation history, extracted data |

### 🌐 User Interfaces

| File | Purpose | Key Features |
|------|---------|--------------|
| `main.py` | CLI interface | Command-line task execution, simple menu system |
| `gradio_ui.py` | Web interface | Full-featured UI, batch processing, session management |
| `launch_gradio.py` | UI launcher | Environment validation, dependency checking, smart setup |
| `gradio_extensions.py` | Advanced UI features | Analytics dashboard, Excel export, performance metrics |

### 🧰 Tools & Utilities

| File | Purpose | Key Features |
|------|---------|--------------|
| `toolkit/web_toolkit.py` | Batch processing | Parallel execution, CSV/JSON export, result aggregation |
| `examples/demo_tasks.py` | Example workflows | News extraction, form filling, ready-to-run demos |
| `examples/utils_examples.py` | Utility examples | Project setup, batch demos, cleanup scripts |

### 🧪 Testing Suite

| File | Purpose | Key Features |
|------|---------|--------------|
| `tests/test_playwright.py` | Browser smoke tests | Playwright installation validation |
| `tests/test_playwright_manager.py` | Browser manager tests | Async operations, element extraction |
| `tests/test_web_browsing_agent.py` | Agent pipeline tests | Full workflow validation, error handling |
| `tests/test_web_toolkit.py` | Export functionality | CSV/JSON export validation |
| `tests/test_config.py` | Configuration tests | Environment variable handling |

## 🚀 Getting Started Workflows

### 1. **Initial Setup**
```bash
# Clone/download project
git clone <repository-url>
cd playwright_langgraph_agent

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements_gradio.txt
playwright install

# Setup environment
echo "OPENAI_API_KEY=your-key-here" > .env
```

### 2. **Quick CLI Test**
```bash
python main.py
# Enter URL and task when prompted
```

### 3. **Launch Web Interface**
```bash
python launch_gradio.py
# Opens browser to http://localhost:7860
```

### 4. **Run Tests**
```bash
pytest tests/
```

### 5. **Try Examples**
```bash
python examples/demo_tasks.py
python examples/utils_examples.py
```

## 🎯 Key Entry Points

### **For End Users**
- **🌐 Web Interface**: `python launch_gradio.py` → http://localhost:7860
- **💻 CLI Interface**: `python main.py`
- **📚 Examples**: `python examples/demo_tasks.py`

### **For Developers**
- **🧪 Testing**: `pytest tests/`
- **🔧 Agent Logic**: `agent/web_browsing_agent.py`
- **🌐 Browser Ops**: `browser/playwright_manager.py`
- **📊 UI Features**: `gradio_ui.py`

### **For DevOps**
- **🚀 Deployment**: `launch_gradio.py`
- **📋 Dependencies**: `requirements_gradio.txt`
- **⚙️ Configuration**: `config.py`

## 🔧 Configuration Files

### **Environment Variables (`.env`)**
```bash
# Required
OPENAI_API_KEY=sk-your-openai-api-key-here

# Optional
DEFAULT_MODEL=gpt-4
DEBUG=false
LOG_LEVEL=INFO
```

### **Core Dependencies (`requirements.txt`)**
```
langgraph==0.4.8
langchain==0.1.17
langchain-openai==0.1.7
playwright>=1.43.0
pydantic>=2.6
python-dotenv>=1.0
tqdm>=4.66
nest_asyncio>=1.6
```

### **Extended Dependencies (`requirements_gradio.txt`)**
```
-r requirements.txt
gradio>=4.0.0
pandas>=2.0.0
pillow>=10.0.0
matplotlib>=3.7.0
plotly>=5.15.0
openpyxl>=3.1.0
rich>=13.0.0
```

## 📊 Data Flow Architecture

```
🌐 User Input (URL + Task)
    ↓
⚙️ Config & State Initialization
    ↓
🤖 WebBrowsingAgent (LangGraph)
    ↓
🌐 PlaywrightManager (Browser Ops)
    ↓
📊 Data Extraction & Processing
    ↓
💾 Results Storage & Export
    ↓
📈 Analytics & Visualization
```

## 🔄 Execution Flows

### **Single Task Flow**
1. **Input**: URL, task description, task type
2. **Initialize**: Browser, agent, state
3. **Navigate**: Load target page
4. **Analyze**: LLM analyzes page content
5. **Execute**: Extract/interact/search based on task
6. **Complete**: Save results, screenshot, cleanup

### **Batch Processing Flow**
1. **Input**: JSON array of tasks
2. **Parse**: Validate and queue tasks
3. **Execute**: Parallel processing with concurrency control
4. **Aggregate**: Collect all results
5. **Export**: Save to JSON/CSV with metadata
6. **Analyze**: Generate batch statistics

### **Analytics Flow**
1. **Collect**: Load historical data from results/
2. **Process**: Calculate metrics and trends
3. **Visualize**: Generate charts and graphs
4. **Export**: Create comprehensive reports

## 🛠️ Development Patterns

### **Adding New Features**
1. **State**: Update `state.py` with new fields
2. **Agent**: Add logic to `web_browsing_agent.py`
3. **UI**: Extend `gradio_ui.py` with new components
4. **Tests**: Add validation in `tests/`
5. **Docs**: Update README files

### **Customizing Tasks**
1. **Define**: New task type in agent routing
2. **Implement**: Handler method in agent class
3. **UI**: Add to task type dropdown
4. **Test**: Create validation tests

### **Extending Exports**
1. **Format**: Add new export function in `toolkit/`
2. **UI**: Add export option in Gradio interface
3. **Analytics**: Update analytics to handle new format

## 🔐 Security & Best Practices

### **API Key Management**
- ✅ Store in `.env` file (gitignored)
- ✅ Runtime validation and error handling
- ✅ Optional environment variable fallback

### **Data Privacy**
- ✅ Local processing only (no data sent to external services except OpenAI)
- ✅ Screenshots stored locally
- ✅ Configurable data retention

### **Error Handling**
- ✅ Graceful failure recovery
- ✅ Detailed logging and debugging
- ✅ User-friendly error messages
- ✅ Retry logic with backoff

## 📦 Deployment Options

### **Local Development**
```bash
python launch_gradio.py  # Development server
```

### **Production Server**
```bash
# With gunicorn (recommended)
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:7860 gradio_ui:demo

# With custom server settings
python gradio_ui.py  # Modify launch() parameters
```

### **Docker Deployment**
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements_gradio.txt .
RUN pip install -r requirements_gradio.txt
RUN playwright install
COPY . .
EXPOSE 7860
CMD ["python", "launch_gradio.py"]
```

## 🎯 Use Cases & Examples

### **Business Intelligence**
- Competitor website monitoring
- Price tracking and analysis
- News and trend aggregation
- Market research automation

### **Quality Assurance**
- Automated website testing
- Form validation workflows
- UI/UX consistency checks
- Performance monitoring

### **Content Management**
- Blog post extraction
- Social media monitoring
- Documentation scraping
- Content migration tasks

### **Research & Academia**
- Paper and citation collection
- Dataset aggregation
- Survey and form automation
- Literature review assistance

This complete project structure provides a robust, scalable foundation for web automation using AI-powered decision making, with both CLI and web interfaces for different user needs.
