# 🕷️ Playwright LangGraph Agent - Gradio UI

A comprehensive web interface for the Playwright LangGraph Agent, providing an intuitive way to automate web browsing tasks using AI.

## 🌟 Features

### 🎯 Single Task Execution
- **Extract Data**: Pull information from web pages automatically
- **Form Interaction**: Fill out forms and click buttons
- **Content Search**: Find specific information on pages
- **Real-time Screenshots**: Visual feedback of agent actions
- **Navigation Tracking**: Step-by-step execution history

### 📊 Batch Processing
- **Parallel Execution**: Process multiple URLs simultaneously
- **Configurable Concurrency**: Control how many tasks run at once
- **JSON Configuration**: Easy task definition format
- **Automatic Export**: Results saved in JSON and CSV formats
- **Progress Tracking**: Real-time batch execution status

### 💾 Session Management
- **History Tracking**: Keep track of all executed tasks
- **Export Options**: Save results in multiple formats
- **Session Persistence**: Maintain state across browser refreshes
- **Clear Controls**: Easy session cleanup

### 🔧 Advanced Features
- **Error Handling**: Graceful failure recovery and reporting
- **API Key Management**: Secure credential handling
- **Responsive Design**: Works on desktop and mobile
- **Dark/Light Themes**: Adaptive UI themes
- **Help Documentation**: Built-in usage examples

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install the extended requirements including Gradio
pip install -r requirements_gradio.txt

# Install Playwright browsers
playwright install
```

### 2. Set Up Environment

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your-api-key-here
DEFAULT_MODEL=gpt-4
```

### 3. Launch the Interface

#### Option A: Use the launcher (Recommended)
```bash
python launch_gradio.py
```

#### Option B: Direct launch
```bash
python gradio_ui.py
```

### 4. Access the Interface

Open your browser and go to: **http://localhost:7860**

## 📖 Usage Guide

### 🎯 Single Tasks

1. **Enter your OpenAI API key** (top of the interface)
2. **Navigate to the "Single Task" tab**
3. **Fill in the task details:**
   - **URL**: Target website
   - **Task**: What you want the agent to do
   - **Type**: Choose extract/interact/search
   - **Form Data**: (Optional) JSON for form interactions
4. **Click "Execute Task"**
5. **Review results** in the output panels

#### Example: Extract News Headlines
- **URL**: `https://news.ycombinator.com`
- **Task**: `Extract the top 10 news headlines and their links`
- **Type**: `extract`

#### Example: Fill a Form
- **URL**: `https://httpbin.org/forms/post`
- **Task**: `Fill out the contact form`
- **Type**: `interact`
- **Form Data**: 
```json
{
  "input[name='custname']": "John Doe",
  "input[name='custemail']": "john@example.com"
}
```

### 📊 Batch Processing

1. **Navigate to the "Batch Processing" tab**
2. **Define your tasks in JSON format:**

```json
[
  {
    "url": "https://news.ycombinator.com",
    "task": "Extract top 5 headlines",
    "task_type": "extract"
  },
  {
    "url": "https://example.com",
    "task": "Get page title and main content",
    "task_type": "extract"
  },
  {
    "url": "https://httpbin.org/forms/post",
    "task": "Fill out the contact form",
    "task_type": "interact",
    "form_data": {
      "input[name='custname']": "Test User",
      "input[name='custemail']": "test@example.com"
    }
  }
]
```

3. **Set concurrency level** (1-10 simultaneous tasks)
4. **Click "Execute Batch"**
5. **Monitor progress** and download results

### 💾 Session Management

1. **Navigate to the "Session Management" tab**
2. **View your session history** with all executed tasks
3. **Export results** in JSON or CSV format
4. **Clear session** when starting fresh

#### Export Formats

**JSON Export**: Complete data with nested structures
```json
[
  {
    "success": true,
    "task": "Extract headlines",
    "url": "https://news.ycombinator.com",
    "extracted_data": {
      "elements": [...],
      "timestamp": "2024-01-15T10:30:00"
    },
    "navigation_history": [...]
  }
]
```

**CSV Export**: Flattened data for spreadsheet analysis
- Nested objects converted to JSON strings
- Easy to import into Excel or Google Sheets

## 🔧 Advanced Configuration

### Form Data Selectors

The agent uses CSS selectors to target form elements. Common patterns:

```json
{
  "#username": "value_for_id",
  ".email-field": "value_for_class",
  "input[name='password']": "value_for_name_attribute",
  "input[type='submit']": "click_action",
  "textarea.description": "multi_line_text"
}
```

### Task Types Explained

#### 📄 Extract
- **Purpose**: Pull data from pages
- **Best for**: Headlines, links, tables, text content
- **Output**: Structured data with elements, text, and metadata

#### 🔗 Interact
- **Purpose**: Fill forms, click buttons, navigate
- **Best for**: Form submissions, button clicks, page interactions
- **Requires**: Form data JSON for target elements

#### 🔍 Search
- **Purpose**: Find specific content on pages
- **Best for**: Locating keywords, phrases, or specific information
- **Output**: Search results with matched content and locations

### Concurrency Guidelines

- **1-2 tasks**: Conservative, good for testing
- **3-5 tasks**: Balanced performance and resource usage
- **6-10 tasks**: High throughput, ensure stable internet connection

## 📁 File Organization

The Gradio UI automatically organizes outputs:

```
results/
├── gradio_sessions/          # Single task exports
│   ├── session_20240115_103000.json
│   └── session_20240115_104500.csv
├── batch_runs/               # Batch processing results
│   └── batch_20240115_105000/
│       ├── results.json      # Complete batch results
│       ├── results.csv       # Flattened batch data
│       └── individual_*.json # Per-task results
└── screenshots/              # Automatic screenshots
    ├── screenshot_20240115_103015.png
    └── screenshot_20240115_103045.png
```

## 🛠️ Troubleshooting

### Common Issues

#### "Agent initialization failed"
- **Check**: OpenAI API key is valid and has credits
- **Solution**: Verify API key in the interface or `.env` file

#### "Navigation failed" 
- **Check**: URL is accessible and correctly formatted
- **Solution**: Test URL in browser first, ensure `https://` prefix

#### "Element not found"
- **Check**: CSS selectors in form data are correct
- **Solution**: Inspect page HTML to verify element selectors

#### "Batch execution stopped"
- **Check**: JSON format is valid
- **Solution**: Validate JSON syntax, ensure all required fields present

#### "Screenshot not captured"
- **Check**: Sufficient disk space and permissions
- **Solution**: Verify `results/screenshots/` directory is writable

### Performance Tips

1. **Start simple**: Test with basic extract tasks first
2. **Use specific selectors**: More precise CSS selectors work better
3. **Limit batch size**: Start with 2-3 tasks, increase gradually
4. **Monitor resources**: Large batches use more memory and CPU
5. **Check logs**: Review console output for detailed error information

### Debug Mode

Enable debug mode by setting environment variable:
```bash
export DEBUG=1
python gradio_ui.py
```

This provides:
- Detailed logging output
- Step-by-step execution traces
- Enhanced error messages
- Performance timing information

## 🔐 Security Notes

- **API Keys**: Never commit API keys to version control
- **Local Only**: Default setup runs on localhost only
- **Public Access**: Use `share=True` in launch settings cautiously
- **Data Privacy**: All processing happens locally unless explicitly shared

## 🤝 Contributing

### Adding New Features

1. **Fork the repository**
2. **Create feature branch**
3. **Add functionality to `gradio_ui.py`**
4. **Update this README**
5. **Test thoroughly**
6. **Submit pull request**

### Custom Task Types

Extend the agent by adding new task types:

```python
# In gradio_ui.py, add to task_type dropdown
task_type = gr.Dropdown(
    label="🔧 Task Type",
    choices=["extract", "interact", "search", "custom"],
    value="extract"
)

# In agent/web_browsing_agent.py, add handler
async def _handle_custom_task(self, state: BrowserState) -> BrowserState:
    # Your custom logic here
    pass
```

## 📚 Examples Library

### E-commerce Data Extraction
```json
{
  "url": "https://example-store.com/products",
  "task": "Extract product names, prices, and availability",
  "task_type": "extract"
}
```

### Social Media Monitoring
```json
{
  "url": "https://twitter.com/username",
  "task": "Extract recent tweet content and engagement metrics",
  "task_type": "extract"
}
```

### Form Automation
```json
{
  "url": "https://forms.example.com/contact",
  "task": "Fill and submit contact form",
  "task_type": "interact",
  "form_data": {
    "#name": "John Doe",
    "#email": "john@example.com",
    "#message": "Automated message via agent",
    "button[type='submit']": "click"
  }
}
```

### Research Data Collection
```json
[
  {
    "url": "https://scholar.google.com/scholar?q=machine+learning",
    "task": "Extract paper titles and authors from search results",
    "task_type": "extract"
  },
  {
    "url": "https://arxiv.org/list/cs.AI/recent",
    "task": "Get recent AI paper abstracts",
    "task_type": "extract"
  }
]
```

## 🆘 Support

- **Documentation**: This README and in-app help tabs
- **Issues**: GitHub repository issues page
- **Community**: Discord/Slack community channels
- **Email**: Technical support contact

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Happy Automating! 🕷️✨**