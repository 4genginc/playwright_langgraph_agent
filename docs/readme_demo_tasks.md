# 📚 Demo Tasks - Complete Pedagogical Guide & Usage Examples

A **comprehensive, production-grade tutorial** for the demo tasks in your Playwright LangGraph Agent project. This guide provides practical examples, learning exercises, and advanced workflows to master autonomous web-browsing agents.

---

## 📁 Project Context

```plaintext
playwright_langgraph_agent/
├── examples/
│   ├── demo_tasks.py         # 🎯 This file - Core demos
│   └── utils_examples.py     # 🛠️ Utility demonstrations
├── agent/
│   └── web_browsing_agent.py # 🤖 The AI agent
├── browser/
│   └── playwright_manager.py # 🌐 Browser automation
├── toolkit/
│   └── web_toolkit.py        # 📊 Batch processing & exports
└── config.py                 # ⚙️ Environment configuration
```

**Role in the System:**
- **Learning Resource**: Hands-on examples for understanding agent capabilities
- **Testing Ground**: Safe environment to experiment with different tasks
- **Template Library**: Reusable patterns for building custom workflows
- **Validation Tool**: Verify your setup works correctly

---

## 🚀 Quickstart: Running Your First Demo

### **Step 1: Environment Setup**
```bash
# Ensure you're in the project directory
cd playwright_langgraph_agent

# Install dependencies if not done yet
pip install -r requirements.txt
playwright install chromium

# Set your OpenAI API key
export OPENAI_API_KEY="sk-your-key-here"
# OR create .env file with: OPENAI_API_KEY=sk-your-key-here
```

### **Step 2: Run the Basic Demo**
```bash
# Run the news extraction demo
python examples/demo_tasks.py
```

### **Step 3: Examine the Results**
```bash
# Check the generated files
ls -la *.json
# Output: demo_news_extraction.json

# View the results
cat demo_news_extraction.json | python -m json.tool
```

---

## 💡 Understanding the Current Demos

### **Demo 1: News Extraction (`demo_news_extraction`)**

**What it does:**
- Navigates to Hacker News
- Uses LLM reasoning to identify headlines
- Extracts top 10 news headlines and their links
- Saves results as structured JSON

**Code Breakdown:**
```python
async def demo_news_extraction():
    # 1. Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    
    # 2. Initialize agent with headless browser
    agent = WebBrowsingAgent(api_key, headless=True)
    
    # 3. Execute extraction task
    result = await agent.execute_task(
        url="https://news.ycombinator.com",           # Target site
        task="Extract the top 10 news headlines and their links",  # LLM instruction
        task_type="extract"                           # Agent mode
    )
    
    # 4. Display and save results
    print("Demo News Extraction Result:")
    print(result)
    export_json([result], "demo_news_extraction.json")
```

**Key Learning Points:**
- **Task Description**: Clear, specific instructions work best
- **Task Type**: "extract" tells the agent to focus on data gathering
- **Headless Mode**: `headless=True` runs browser invisibly for speed
- **Result Structure**: Returns success status, extracted data, and navigation history

### **Demo 2: Form Filling (`demo_form_filling`)**

**What it does:**
- Navigates to a test form (HTTPBin)
- Fills out contact form fields automatically
- Demonstrates interactive capabilities
- Shows form data handling

**Code Breakdown:**
```python
async def demo_form_filling():
    api_key = os.getenv("OPENAI_API_KEY")
    agent = WebBrowsingAgent(api_key, headless=True)
    
    # Define form data to fill
    form_data = {"#name": "Alice", "#email": "alice@example.com"}
    
    result = await agent.execute_task(
        url="https://httpbin.org/forms/post",          # Test form URL
        task="Fill out the contact form",             # LLM instruction
        task_type="interact",                         # Interactive mode
        form_data=form_data                           # Data to input
    )
    
    print("Demo Form Fill Result:")
    print(result)
    export_json([result], "demo_form_fill.json")
```

**Key Learning Points:**
- **Form Data Format**: Use CSS selectors as keys (`#name`, `#email`)
- **Task Type**: "interact" enables form filling and clicking
- **Structured Input**: Pre-defined data ensures consistent testing
- **Test Environment**: HTTPBin provides safe testing endpoints

---

## 🎯 Comprehensive Demo Collection

Let's expand beyond the basic demos with a complete learning progression:

### **Level 1: Basic Operations (Beginner)**

#### **Demo 3: Simple Data Extraction**
```python
async def demo_simple_extraction():
    """Extract basic page information - perfect for beginners"""
    api_key = os.getenv("OPENAI_API_KEY")
    agent = WebBrowsingAgent(api_key, headless=True)
    
    result = await agent.execute_task(
        url="https://example.com",
        task="Extract the page title, main heading, and any paragraph text",
        task_type="extract"
    )
    
    print("📄 Simple Extraction Results:")
    print(f"Success: {result['success']}")
    print(f"Title: {result.get('extracted_data', {}).get('title', 'N/A')}")
    
    export_json([result], "demo_simple_extraction.json")
    return result
```

#### **Demo 4: Link Discovery**
```python
async def demo_link_discovery():
    """Discover and categorize links on a webpage"""
    api_key = os.getenv("OPENAI_API_KEY")
    agent = WebBrowsingAgent(api_key, headless=True)
    
    result = await agent.execute_task(
        url="https://www.python.org",
        task="Find all navigation links and categorize them by type (documentation, downloads, community, etc.)",
        task_type="extract"
    )
    
    print("🔗 Link Discovery Results:")
    extracted = result.get('extracted_data', {})
    elements = extracted.get('elements', [])
    
    links = [elem for elem in elements if elem.get('tag') == 'a']
    print(f"Found {len(links)} links")
    
    export_json([result], "demo_link_discovery.json")
    return result
```

### **Level 2: Interactive Operations (Intermediate)**

#### **Demo 5: Search Functionality**
```python
async def demo_search_interaction():
    """Demonstrate search box interaction"""
    api_key = os.getenv("OPENAI_API_KEY")
    agent = WebBrowsingAgent(api_key, headless=True)
    
    # Search on a documentation site
    result = await agent.execute_task(
        url="https://docs.python.org",
        task="Find the search box and search for 'asyncio', then extract the first 3 search results",
        task_type="interact"
    )
    
    print("🔍 Search Interaction Results:")
    print(f"Success: {result['success']}")
    
    if result['success']:
        nav_history = result.get('navigation_history', [])
        print("Actions taken:")
        for i, action in enumerate(nav_history[-3:], 1):
            print(f"  {i}. {action}")
    
    export_json([result], "demo_search_interaction.json")
    return result
```

#### **Demo 6: Multi-Step Form Workflow**
```python
async def demo_multi_step_form():
    """Handle complex forms with multiple fields and validation"""
    api_key = os.getenv("OPENAI_API_KEY")
    agent = WebBrowsingAgent(api_key, headless=True)
    
    # Comprehensive form data
    form_data = {
        "#name": "John Doe",
        "#email": "john.doe@example.com",
        "#phone": "555-0123",
        "#message": "This is a test message from the Playwright LangGraph Agent.",
        "#subject": "Demo Contact Form Submission"
    }
    
    result = await agent.execute_task(
        url="https://httpbin.org/forms/post",
        task="Fill out all form fields completely and submit the form",
        task_type="interact",
        form_data=form_data
    )
    
    print("📝 Multi-Step Form Results:")
    print(f"Success: {result['success']}")
    
    # Analyze the navigation history for form steps
    nav_history = result.get('navigation_history', [])
    form_actions = [step for step in nav_history if 'form' in step.lower() or 'fill' in step.lower()]
    print(f"Form actions performed: {len(form_actions)}")
    
    export_json([result], "demo_multi_step_form.json")
    return result
```

### **Level 3: Advanced Workflows (Advanced)**

#### **Demo 7: E-commerce Product Research**
```python
async def demo_ecommerce_research():
    """Research products across e-commerce sites"""
    api_key = os.getenv("OPENAI_API_KEY")
    agent = WebBrowsingAgent(api_key, headless=True)
    
    # Research multiple product categories
    product_searches = [
        {
            "url": "https://www.amazon.com",
            "task": "Search for 'bluetooth headphones' and extract the top 3 product names, prices, and ratings",
            "category": "electronics"
        },
        {
            "url": "https://www.etsy.com",
            "task": "Search for 'handmade jewelry' and extract featured product titles and shop names",
            "category": "handmade"
        }
    ]
    
    all_results = []
    
    for search in product_searches:
        print(f"🛍️ Researching {search['category']} products...")
        
        result = await agent.execute_task(
            url=search["url"],
            task=search["task"],
            task_type="search"
        )
        
        result["category"] = search["category"]
        result["search_intent"] = search["task"]
        all_results.append(result)
        
        # Respectful delay between requests
        await asyncio.sleep(3)
    
    print(f"🎯 Product Research Complete: {len(all_results)} categories analyzed")
    export_json(all_results, "demo_ecommerce_research.json")
    return all_results
```

#### **Demo 8: News Aggregation from Multiple Sources**
```python
async def demo_news_aggregation():
    """Aggregate news from multiple sources with analysis"""
    api_key = os.getenv("OPENAI_API_KEY")
    agent = WebBrowsingAgent(api_key, headless=True)
    
    news_sources = [
        {
            "url": "https://news.ycombinator.com",
            "task": "Extract top 5 technology news headlines with scores and comment counts",
            "source": "Hacker News",
            "category": "technology"
        },
        {
            "url": "https://www.bbc.com/news/technology",
            "task": "Extract the main technology news headlines and brief descriptions",
            "source": "BBC News",
            "category": "technology"
        },
        {
            "url": "https://techcrunch.com",
            "task": "Extract latest startup and tech company news headlines",
            "source": "TechCrunch",
            "category": "startup"
        }
    ]
    
    aggregated_news = []
    
    for source in news_sources:
        print(f"📰 Collecting from {source['source']}...")
        
        result = await agent.execute_task(
            url=source["url"],
            task=source["task"],
            task_type="extract"
        )
        
        # Enrich with metadata
        enriched_result = {
            **result,
            "source_name": source["source"],
            "category": source["category"],
            "collection_timestamp": datetime.now().isoformat()
        }
        
        aggregated_news.append(enriched_result)
        
        # Respectful delay
        await asyncio.sleep(2)
    
    # Generate summary report
    successful_sources = [news for news in aggregated_news if news.get('success')]
    
    summary = {
        "aggregation_timestamp": datetime.now().isoformat(),
        "total_sources": len(news_sources),
        "successful_sources": len(successful_sources),
        "total_headlines": sum(
            len(news.get('extracted_data', {}).get('elements', []))
            for news in successful_sources
        ),
        "sources_processed": [news["source_name"] for news in successful_sources]
    }
    
    # Save detailed results and summary
    export_json(aggregated_news, "demo_news_aggregation_detailed.json")
    export_json(summary, "demo_news_aggregation_summary.json")
    
    print(f"📊 News Aggregation Summary:")
    print(f"  Sources processed: {summary['successful_sources']}/{summary['total_sources']}")
    print(f"  Total headlines collected: {summary['total_headlines']}")
    
    return aggregated_news
```

### **Level 4: Specialized Use Cases (Expert)**

#### **Demo 9: Competitive Analysis Dashboard**
```python
async def demo_competitive_analysis():
    """Perform competitive analysis across multiple websites"""
    api_key = os.getenv("OPENAI_API_KEY")
    agent = WebBrowsingAgent(api_key, headless=True)
    
    competitors = [
        {
            "name": "Competitor A",
            "url": "https://example-competitor-a.com",
            "analysis_points": [
                "pricing information",
                "key features",
                "contact information",
                "company size indicators"
            ]
        },
        {
            "name": "Competitor B", 
            "url": "https://example-competitor-b.com",
            "analysis_points": [
                "service offerings",
                "team information",
                "customer testimonials",
                "technology stack"
            ]
        }
    ]
    
    competitive_data = []
    
    for competitor in competitors:
        print(f"🔍 Analyzing {competitor['name']}...")
        
        analysis_task = f"""
        Analyze this website for competitive intelligence:
        - Extract {', '.join(competitor['analysis_points'])}
        - Identify unique selling propositions
        - Note any special offers or promotions
        - Assess overall website quality and user experience
        """
        
        result = await agent.execute_task(
            url=competitor["url"],
            task=analysis_task,
            task_type="extract"
        )
        
        competitive_analysis = {
            "competitor_name": competitor["name"],
            "analysis_date": datetime.now().isoformat(),
            "url": competitor["url"],
            "analysis_points": competitor["analysis_points"],
            "raw_result": result,
            "success": result.get("success", False)
        }
        
        competitive_data.append(competitive_analysis)
        
        # Take screenshot for visual reference
        if result.get("screenshot"):
            print(f"📸 Screenshot saved: {result['screenshot']}")
        
        await asyncio.sleep(5)  # Longer delay for respectful analysis
    
    # Generate competitive intelligence report
    intelligence_report = {
        "report_generated": datetime.now().isoformat(),
        "competitors_analyzed": len(competitors),
        "successful_analyses": sum(1 for comp in competitive_data if comp["success"]),
        "competitive_insights": competitive_data,
        "methodology": "Automated web analysis using LLM-guided browser automation"
    }
    
    export_json(intelligence_report, "demo_competitive_analysis.json")
    
    print(f"📊 Competitive Analysis Complete:")
    print(f"  Competitors analyzed: {intelligence_report['successful_analyses']}/{intelligence_report['competitors_analyzed']}")
    
    return intelligence_report
```

#### **Demo 10: Documentation Mining**
```python
async def demo_documentation_mining():
    """Mine technical documentation for specific information"""
    api_key = os.getenv("OPENAI_API_KEY")
    agent = WebBrowsingAgent(api_key, headless=True)
    
    documentation_sites = [
        {
            "name": "Python Asyncio Docs",
            "url": "https://docs.python.org/3/library/asyncio.html",
            "mining_task": "Extract all asyncio function names and their brief descriptions"
        },
        {
            "name": "Playwright Python Docs",
            "url": "https://playwright.dev/python/",
            "mining_task": "Extract the main API methods and their usage examples"
        },
        {
            "name": "LangChain Docs",
            "url": "https://python.langchain.com/docs/",
            "mining_task": "Find and extract the core concepts and their explanations"
        }
    ]
    
    documentation_data = []
    
    for doc_site in documentation_sites:
        print(f"📚 Mining {doc_site['name']}...")
        
        result = await agent.execute_task(
            url=doc_site["url"],
            task=doc_site["mining_task"],
            task_type="extract"
        )
        
        doc_analysis = {
            "documentation_source": doc_site["name"],
            "mining_timestamp": datetime.now().isoformat(),
            "url": doc_site["url"],
            "mining_objective": doc_site["mining_task"],
            "extraction_result": result,
            "content_quality": "high" if result.get("success") else "extraction_failed"
        }
        
        documentation_data.append(doc_analysis)
        await asyncio.sleep(3)
    
    # Create knowledge base summary
    knowledge_base = {
        "kb_created": datetime.now().isoformat(),
        "sources_mined": len(documentation_sites),
        "successful_extractions": sum(1 for doc in documentation_data if doc["extraction_result"].get("success")),
        "documentation_insights": documentation_data,
        "potential_applications": [
            "API reference generation",
            "Code example extraction",
            "Tutorial content creation",
            "Technical knowledge aggregation"
        ]
    }
    
    export_json(knowledge_base, "demo_documentation_mining.json")
    
    print(f"📖 Documentation Mining Summary:")
    print(f"  Sources processed: {knowledge_base['successful_extractions']}/{knowledge_base['sources_mined']}")
    
    return knowledge_base
```

---

## 🛠️ Utility Functions for Enhanced Demos

### **Demo Result Analysis**
```python
def analyze_demo_results(result_file):
    """Analyze and summarize demo results"""
    from utils import load_json
    
    try:
        data = load_json(result_file)
        
        if isinstance(data, list):
            # Multiple results
            successful = sum(1 for r in data if r.get('success'))
            total = len(data)
            
            print(f"📊 Batch Results Analysis for {result_file}:")
            print(f"  Success Rate: {successful}/{total} ({successful/total:.1%})")
            
            # Analyze extraction data
            total_elements = sum(
                len(r.get('extracted_data', {}).get('elements', []))
                for r in data if r.get('success')
            )
            print(f"  Total Elements Extracted: {total_elements}")
            
        else:
            # Single result
            print(f"📊 Single Result Analysis for {result_file}:")
            print(f"  Success: {data.get('success')}")
            print(f"  URL: {data.get('final_url', 'N/A')}")
            
            extracted = data.get('extracted_data', {})
            if extracted:
                elements = extracted.get('elements', [])
                print(f"  Elements Extracted: {len(elements)}")
                
                # Show element types
                element_types = {}
                for elem in elements:
                    tag = elem.get('tag', 'unknown')
                    element_types[tag] = element_types.get(tag, 0) + 1
                
                print(f"  Element Types: {dict(element_types)}")
    
    except Exception as e:
        print(f"❌ Failed to analyze {result_file}: {e}")
```

### **Demo Comparison Tool**
```python
def compare_demo_results(file1, file2):
    """Compare results from two different demo runs"""
    from utils import load_json
    
    try:
        data1 = load_json(file1)
        data2 = load_json(file2)
        
        print(f"🔍 Comparing {file1} vs {file2}")
        
        # Compare success rates
        success1 = data1.get('success', False)
        success2 = data2.get('success', False)
        
        print(f"Success: {file1}={success1}, {file2}={success2}")
        
        # Compare element counts
        elements1 = len(data1.get('extracted_data', {}).get('elements', []))
        elements2 = len(data2.get('extracted_data', {}).get('elements', []))
        
        print(f"Elements Extracted: {file1}={elements1}, {file2}={elements2}")
        
        # Compare navigation steps
        nav1 = len(data1.get('navigation_history', []))
        nav2 = len(data2.get('navigation_history', []))
        
        print(f"Navigation Steps: {file1}={nav1}, {file2}={nav2}")
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
```

---

## 🎓 Learning Exercises & Challenges

### **Exercise 1: Custom News Site**
**Objective**: Adapt the news extraction demo to work with a different news website.

**Your Task**: 
1. Choose a news website (e.g., Reuters, Associated Press)
2. Modify `demo_news_extraction()` to target your chosen site
3. Adjust the task description for the site's structure
4. Compare results with the original Hacker News demo

**Template**:
```python
async def exercise_custom_news_site():
    """YOUR IMPLEMENTATION HERE"""
    api_key = os.getenv("OPENAI_API_KEY")
    agent = WebBrowsingAgent(api_key, headless=True)
    
    # TODO: Choose your target news site
    target_url = "https://your-chosen-news-site.com"
    
    # TODO: Customize the extraction task
    extraction_task = "Your custom extraction instruction"
    
    result = await agent.execute_task(
        url=target_url,
        task=extraction_task,
        task_type="extract"
    )
    
    # TODO: Add your analysis
    export_json([result], "exercise_custom_news.json")
    return result
```

### **Exercise 2: Multi-Page Navigation**
**Objective**: Create a demo that navigates through multiple pages on a website.

**Your Task**:
1. Start on a website's homepage
2. Navigate to a specific section (e.g., "About", "Products")
3. Extract information from the target page
4. Return to homepage and navigate to another section

### **Exercise 3: Form Validation Testing**
**Objective**: Test form validation by intentionally providing invalid data.

**Your Task**:
1. Use a form with validation (email format, required fields)
2. Try submitting with missing or invalid data
3. Capture and analyze any error messages
4. Then submit with valid data

### **Challenge 1: Social Media Monitoring**
**Objective**: Monitor social media platforms for specific keywords or hashtags.

**Requirements**:
- Search for specific terms on multiple platforms
- Extract relevant posts/tweets
- Analyze sentiment or engagement metrics
- Generate a monitoring report

### **Challenge 2: Job Market Analysis**
**Objective**: Analyze job postings across multiple job boards.

**Requirements**:
- Search for specific job titles/keywords
- Extract job details (salary, location, requirements)
- Compare opportunities across different platforms
- Generate market analysis report

---

## 🔧 Advanced Configuration & Customization

### **Custom Demo Runner**
```python
class DemoRunner:
    """Advanced demo execution with monitoring and retry logic"""
    
    def __init__(self, api_key, headless=True):
        self.api_key = api_key
        self.headless = headless
        self.results = []
        
    async def run_demo(self, demo_func, max_retries=2):
        """Run a demo with retry logic and monitoring"""
        for attempt in range(max_retries + 1):
            try:
                print(f"🚀 Running {demo_func.__name__} (attempt {attempt + 1})")
                
                start_time = datetime.now()
                result = await demo_func()
                end_time = datetime.now()
                
                # Add execution metadata
                execution_info = {
                    "demo_name": demo_func.__name__,
                    "execution_time": (end_time - start_time).total_seconds(),
                    "attempt_number": attempt + 1,
                    "timestamp": start_time.isoformat(),
                    "success": result.get('success', False) if isinstance(result, dict) else True
                }
                
                self.results.append(execution_info)
                
                print(f"✅ {demo_func.__name__} completed in {execution_info['execution_time']:.2f}s")
                return result
                
            except Exception as e:
                print(f"❌ {demo_func.__name__} failed (attempt {attempt + 1}): {e}")
                if attempt == max_retries:
                    print(f"🔥 {demo_func.__name__} failed after {max_retries + 1} attempts")
                    return {"success": False, "error": str(e)}
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    def generate_execution_report(self):
        """Generate a comprehensive execution report"""
        if not self.results:
            return {"error": "No demo executions recorded"}
        
        total_demos = len(self.results)
        successful_demos = sum(1 for r in self.results if r.get('success'))
        total_time = sum(r.get('execution_time', 0) for r in self.results)
        
        report = {
            "execution_summary": {
                "total_demos_run": total_demos,
                "successful_demos": successful_demos,
                "success_rate": successful_demos / total_demos if total_demos > 0 else 0,
                "total_execution_time": total_time,
                "average_execution_time": total_time / total_demos if total_demos > 0 else 0
            },
            "demo_details": self.results,
            "report_generated": datetime.now().isoformat()
        }
        
        return report

# Usage example
async def run_all_demos():
    """Run all demos with monitoring"""
    api_key = os.getenv("OPENAI_API_KEY")
    runner = DemoRunner(api_key, headless=True)
    
    demos = [
        demo_news_extraction,
        demo_form_filling,
        demo_simple_extraction,
        demo_link_discovery
    ]
    
    for demo in demos:
        await runner.run_demo(demo)
        await asyncio.sleep(1)  # Brief pause between demos
    
    # Generate and save execution report
    report = runner.generate_execution_report()
    export_json(report, f"demo_execution_report_{timestamp_str()}.json")
    
    print("\n📊 Demo Execution Summary:")
    print(f"  Total Demos: {report['execution_summary']['total_demos_run']}")
    print(f"  Success Rate: {report['execution_summary']['success_rate']:.1%}")
    print(f"  Total Time: {report['execution_summary']['total_execution_time']:.2f}s")
```

### **Interactive Demo Selector**
```python
def interactive_demo_selector():
    """Interactive CLI for selecting and running demos"""
    available_demos = {
        "1": ("Basic News Extraction", demo_news_extraction),
        "2": ("Form Filling", demo_form_filling),
        "3": ("Simple Extraction", demo_simple_extraction),
        "4": ("Link Discovery", demo_link_discovery),
        "5": ("E-commerce Research", demo_ecommerce_research),
        "6": ("News Aggregation", demo_news_aggregation),
        "7": ("Competitive Analysis", demo_competitive_analysis),
        "8": ("Documentation Mining", demo_documentation_mining),
        "0": ("Run All Demos", None)
    }
    
    print_banner("🎯 Demo Selector", char="=", width=50)
    print("Available demos:")
    
    for key, (name, _) in available_demos.items():
        print(f"  {key}. {name}")
    
    while True:
        choice = input("\nSelect a demo to run (or 'q' to quit): ").strip()
        
        if choice.lower() == 'q':
            print("👋 Goodbye!")
            break
            
        if choice in available_demos:
            demo_name, demo_func = available_demos[choice]
            
            if choice == "0":
                # Run all demos
                asyncio.run(run_all_demos())
            else:
                print(f"\n🚀 Running: {demo_name}")
                try:
                    asyncio.run(demo_func())
                    print(f"✅ {demo_name} completed successfully!")
                except Exception as e:
                    print(f"❌ {demo_name} failed: {e}")
        else:
            print("❌ Invalid selection. Please try again.")

# Add to main execution
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_demo_selector()
    else:
        # Run default demo
        asyncio.run(demo_news_extraction())
```

---

## 🧪 Testing & Validation

### **Demo Test Suite**
```python
import pytest
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_demo_news_extraction():
    """Test the news extraction demo with mocked agent"""
    with patch('examples.demo_tasks.WebBrowsingAgent') as mock_agent_class:
        # Setup mock
        mock_agent = AsyncMock()
        mock_agent.execute_task.return_value = {
            "success": True,
            "extracted_data": {"elements": [{"tag": "h1", "text": "Test Headline"}]},
            "navigation_history": ["Navigated to site", "Extracted headlines"],
            "final_url": "https://news.ycombinator.com"
        }
        mock_agent_class.return_value = mock_agent
        
        # Run demo
        result = await demo_news_extraction()
        
        # Verify mock was called correctly
        mock_agent.execute_task.assert_called_once()
        call_args = mock_agent.execute_task.call_args
        
        assert call_args[1]['url'] == "https://news.ycombinator.com"
        assert "headlines" in call_args[1]['task'].lower()
        assert call_args[1]['task_type'] == "extract"

@pytest.mark.asyncio
async def test_demo_form_filling():
    """Test the form filling demo with mocked agent"""
    with patch('examples.demo_tasks.WebBrowsingAgent') as mock_agent_class:
        mock_agent = AsyncMock()
        mock_agent.execute_task.return_value = {
            "success": True,
            "extracted_data": {},
            "navigation_history": ["Form filled successfully"],
            "final_url": "https://httpbin.org/forms/post"
        }
        mock_agent_class.return_value = mock_agent
        
        result = await demo_form_filling()
        
        # Verify form data was passed
        call_args = mock_agent.execute_task.call_args
        assert call_args[1]['task_type'] == "interact"
        assert 'form_data' in call_args[1]
        assert '#name' in call_args[1]['form_data']

def test_demo_environment_validation():
    """Test that demo environment is properly configured"""
    import os
    from config import load_env
    
    # Test environment loading
    load_env()
    
    # Check for required API key
    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key is not None, "OPENAI_API_KEY must be set for demos"
    assert api_key.startswith("sk-"), "API key should start with 'sk-'"

@pytest.mark.integration
@pytest.mark.asyncio
async def test_demo_integration_simple():
    """Integration test with a simple, reliable demo"""
    import os
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set - skipping integration test")
    
    # Run a simple demo that should work reliably
    result = await demo_simple_extraction()
    
    # Basic assertions
    assert isinstance(result, dict)
    assert "success" in result
    assert "extracted_data" in result
    assert "navigation_history" in result
```

### **Demo Performance Benchmarking**
```python
import time
import statistics
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class DemoPerformanceMetric:
    demo_name: str
    execution_time: float
    success: bool
    elements_extracted: int
    navigation_steps: int
    error_message: str = ""

class DemoPerformanceBenchmark:
    """Benchmark demo performance across multiple runs"""
    
    def __init__(self):
        self.metrics: List[DemoPerformanceMetric] = []
    
    async def benchmark_demo(self, demo_func, iterations=3):
        """Benchmark a demo function multiple times"""
        print(f"🏃 Benchmarking {demo_func.__name__} ({iterations} iterations)")
        
        for i in range(iterations):
            start_time = time.time()
            
            try:
                result = await demo_func()
                execution_time = time.time() - start_time
                
                # Extract metrics from result
                if isinstance(result, dict):
                    success = result.get('success', False)
                    extracted_data = result.get('extracted_data', {})
                    elements_extracted = len(extracted_data.get('elements', []))
                    navigation_steps = len(result.get('navigation_history', []))
                    error_message = result.get('error', '')
                else:
                    success = True
                    elements_extracted = 0
                    navigation_steps = 0
                    error_message = ''
                
                metric = DemoPerformanceMetric(
                    demo_name=demo_func.__name__,
                    execution_time=execution_time,
                    success=success,
                    elements_extracted=elements_extracted,
                    navigation_steps=navigation_steps,
                    error_message=error_message
                )
                
                self.metrics.append(metric)
                print(f"  Run {i+1}: {execution_time:.2f}s ({'✅' if success else '❌'})")
                
            except Exception as e:
                execution_time = time.time() - start_time
                metric = DemoPerformanceMetric(
                    demo_name=demo_func.__name__,
                    execution_time=execution_time,
                    success=False,
                    elements_extracted=0,
                    navigation_steps=0,
                    error_message=str(e)
                )
                self.metrics.append(metric)
                print(f"  Run {i+1}: {execution_time:.2f}s ❌ {e}")
            
            # Brief pause between runs
            if i < iterations - 1:
                await asyncio.sleep(2)
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance analysis"""
        if not self.metrics:
            return {"error": "No metrics collected"}
        
        # Group metrics by demo name
        demo_groups = {}
        for metric in self.metrics:
            if metric.demo_name not in demo_groups:
                demo_groups[metric.demo_name] = []
            demo_groups[metric.demo_name].append(metric)
        
        demo_analyses = {}
        
        for demo_name, metrics in demo_groups.items():
            successful_metrics = [m for m in metrics if m.success]
            
            if successful_metrics:
                execution_times = [m.execution_time for m in successful_metrics]
                elements_extracted = [m.elements_extracted for m in successful_metrics]
                navigation_steps = [m.navigation_steps for m in successful_metrics]
                
                analysis = {
                    "total_runs": len(metrics),
                    "successful_runs": len(successful_metrics),
                    "success_rate": len(successful_metrics) / len(metrics),
                    "performance": {
                        "avg_execution_time": statistics.mean(execution_times),
                        "min_execution_time": min(execution_times),
                        "max_execution_time": max(execution_times),
                        "std_execution_time": statistics.stdev(execution_times) if len(execution_times) > 1 else 0
                    },
                    "extraction_metrics": {
                        "avg_elements_extracted": statistics.mean(elements_extracted),
                        "avg_navigation_steps": statistics.mean(navigation_steps)
                    },
                    "reliability_score": len(successful_metrics) / len(metrics) * 100
                }
            else:
                analysis = {
                    "total_runs": len(metrics),
                    "successful_runs": 0,
                    "success_rate": 0,
                    "reliability_score": 0,
                    "common_errors": [m.error_message for m in metrics if m.error_message]
                }
            
            demo_analyses[demo_name] = analysis
        
        # Overall summary
        all_successful = sum(1 for m in self.metrics if m.success)
        all_total = len(self.metrics)
        
        report = {
            "benchmark_timestamp": datetime.now().isoformat(),
            "overall_summary": {
                "total_demo_runs": all_total,
                "successful_runs": all_successful,
                "overall_success_rate": all_successful / all_total if all_total > 0 else 0,
                "demos_benchmarked": len(demo_groups)
            },
            "demo_analyses": demo_analyses,
            "raw_metrics": [
                {
                    "demo_name": m.demo_name,
                    "execution_time": m.execution_time,
                    "success": m.success,
                    "elements_extracted": m.elements_extracted,
                    "navigation_steps": m.navigation_steps,
                    "error_message": m.error_message
                }
                for m in self.metrics
            ]
        }
        
        return report

# Usage example
async def run_performance_benchmark():
    """Run comprehensive performance benchmark"""
    print_banner("🏁 Demo Performance Benchmark")
    
    benchmark = DemoPerformanceBenchmark()
    
    # Benchmark core demos
    core_demos = [
        demo_simple_extraction,
        demo_news_extraction,
        demo_form_filling,
        demo_link_discovery
    ]
    
    for demo in core_demos:
        await benchmark.benchmark_demo(demo, iterations=3)
        print()  # Add spacing between demos
    
    # Generate and save report
    report = benchmark.generate_performance_report()
    export_json(report, f"demo_performance_report_{timestamp_str()}.json")
    
    # Print summary
    print("📊 Performance Benchmark Summary:")
    print(f"  Overall Success Rate: {report['overall_summary']['overall_success_rate']:.1%}")
    print(f"  Total Demo Runs: {report['overall_summary']['total_demo_runs']}")
    
    print("\n🎯 Demo-Specific Results:")
    for demo_name, analysis in report['demo_analyses'].items():
        if analysis.get('performance'):
            avg_time = analysis['performance']['avg_execution_time']
            success_rate = analysis['success_rate']
            print(f"  {demo_name}: {avg_time:.2f}s avg, {success_rate:.1%} success")
        else:
            print(f"  {demo_name}: Failed all runs")
```

---

## 🚀 Production-Ready Demo Patterns

### **Batch Demo Execution**
```python
class BatchDemoExecutor:
    """Execute multiple demos in batch with comprehensive logging"""
    
    def __init__(self, api_key, output_dir="demo_results"):
        self.api_key = api_key
        self.output_dir = output_dir
        ensure_dir(output_dir)
        
        # Setup logging
        log_file = f"{output_dir}/batch_execution_{timestamp_str()}.log"
        setup_basic_logging("INFO", log_file)
        self.logger = logging.getLogger(__name__)
    
    async def execute_demo_batch(self, demo_configs):
        """Execute a batch of demo configurations"""
        batch_id = timestamp_str()
        batch_results = []
        
        self.logger.info(f"Starting batch execution {batch_id} with {len(demo_configs)} demos")
        
        for i, config in enumerate(demo_configs, 1):
            demo_name = config['name']
            demo_func = config['function']
            
            self.logger.info(f"Executing demo {i}/{len(demo_configs)}: {demo_name}")
            print(f"🔄 Running {demo_name} ({i}/{len(demo_configs)})")
            
            start_time = datetime.now()
            
            try:
                # Execute the demo
                result = await demo_func()
                
                execution_time = (datetime.now() - start_time).total_seconds()
                
                # Enrich result with batch metadata
                batch_result = {
                    "batch_id": batch_id,
                    "demo_name": demo_name,
                    "execution_order": i,
                    "execution_timestamp": start_time.isoformat(),
                    "execution_time_seconds": execution_time,
                    "demo_result": result,
                    "success": result.get('success', False) if isinstance(result, dict) else True
                }
                
                batch_results.append(batch_result)
                
                # Save individual result
                individual_file = f"{self.output_dir}/{batch_id}_{i:02d}_{demo_name}.json"
                export_json(batch_result, individual_file)
                
                self.logger.info(f"Demo {demo_name} completed in {execution_time:.2f}s")
                
                # Respectful delay between demos
                if i < len(demo_configs):
                    await asyncio.sleep(2)
                    
            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds()
                error_result = {
                    "batch_id": batch_id,
                    "demo_name": demo_name,
                    "execution_order": i,
                    "execution_timestamp": start_time.isoformat(),
                    "execution_time_seconds": execution_time,
                    "success": False,
                    "error": str(e),
                    "demo_result": None
                }
                
                batch_results.append(error_result)
                self.logger.error(f"Demo {demo_name} failed: {e}")
        
        # Generate batch summary
        successful_demos = sum(1 for r in batch_results if r.get('success'))
        total_time = sum(r.get('execution_time_seconds', 0) for r in batch_results)
        
        batch_summary = {
            "batch_id": batch_id,
            "batch_completed": datetime.now().isoformat(),
            "total_demos": len(demo_configs),
            "successful_demos": successful_demos,
            "success_rate": successful_demos / len(demo_configs),
            "total_execution_time": total_time,
            "average_demo_time": total_time / len(demo_configs),
            "results": batch_results
        }
        
        # Save batch summary
        summary_file = f"{self.output_dir}/{batch_id}_batch_summary.json"
        export_json(batch_summary, summary_file)
        
        self.logger.info(f"Batch execution completed: {successful_demos}/{len(demo_configs)} successful")
        
        return batch_summary

# Predefined demo configurations
DEMO_CONFIGURATIONS = [
    {"name": "simple_extraction", "function": demo_simple_extraction},
    {"name": "news_extraction", "function": demo_news_extraction},
    {"name": "form_filling", "function": demo_form_filling},
    {"name": "link_discovery", "function": demo_link_discovery},
    {"name": "search_interaction", "function": demo_search_interaction},
]

ADVANCED_DEMO_CONFIGURATIONS = [
    {"name": "ecommerce_research", "function": demo_ecommerce_research},
    {"name": "news_aggregation", "function": demo_news_aggregation},
    {"name": "competitive_analysis", "function": demo_competitive_analysis},
    {"name": "documentation_mining", "function": demo_documentation_mining},
]

async def run_standard_demo_suite():
    """Run the standard demo suite"""
    api_key = os.getenv("OPENAI_API_KEY")
    executor = BatchDemoExecutor(api_key)
    
    print_banner("🚀 Standard Demo Suite Execution")
    summary = await executor.execute_demo_batch(DEMO_CONFIGURATIONS)
    
    print(f"\n📊 Standard Suite Results:")
    print(f"  Success Rate: {summary['success_rate']:.1%}")
    print(f"  Total Time: {summary['total_execution_time']:.2f}s")
    print(f"  Results saved to: {executor.output_dir}")
    
    return summary

async def run_advanced_demo_suite():
    """Run the advanced demo suite"""
    api_key = os.getenv("OPENAI_API_KEY")
    executor = BatchDemoExecutor(api_key, output_dir="advanced_demo_results")
    
    print_banner("🔬 Advanced Demo Suite Execution")
    summary = await executor.execute_demo_batch(ADVANCED_DEMO_CONFIGURATIONS)
    
    print(f"\n📊 Advanced Suite Results:")
    print(f"  Success Rate: {summary['success_rate']:.1%}")
    print(f"  Total Time: {summary['total_execution_time']:.2f}s")
    print(f"  Results saved to: {executor.output_dir}")
    
    return summary
```

### **Demo Result Dashboard**
```python
def generate_demo_dashboard(results_directory="demo_results"):
    """Generate an HTML dashboard from demo results"""
    import glob
    import json
    from datetime import datetime
    
    # Collect all result files
    result_files = glob.glob(f"{results_directory}/*_batch_summary.json")
    
    if not result_files:
        print(f"❌ No batch summary files found in {results_directory}")
        return
    
    # Load and analyze results
    all_batches = []
    for file_path in result_files:
        try:
            with open(file_path, 'r') as f:
                batch_data = json.load(f)
                all_batches.append(batch_data)
        except Exception as e:
            print(f"⚠️ Failed to load {file_path}: {e}")
    
    if not all_batches:
        print("❌ No valid batch data found")
        return
    
    # Generate HTML dashboard
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Demo Results Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
            .header {{ text-align: center; color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 20px; }}
            .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
            .summary-card {{ background: #f9f9f9; padding: 15px; border-radius: 8px; border-left: 4px solid #4CAF50; }}
            .batch-section {{ margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }}
            .demo-result {{ margin: 10px 0; padding: 10px; background: #f0f0f0; border-radius: 4px; }}
            .success {{ color: #4CAF50; font-weight: bold; }}
            .failure {{ color: #f44336; font-weight: bold; }}
            .chart-container {{ margin: 20px 0; text-align: center; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎯 Demo Results Dashboard</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
    """
    
    # Overall statistics
    total_batches = len(all_batches)
    total_demos = sum(batch['total_demos'] for batch in all_batches)
    total_successful = sum(batch['successful_demos'] for batch in all_batches)
    overall_success_rate = total_successful / total_demos if total_demos > 0 else 0
    
    html_content += f"""
            <div class="summary-grid">
                <div class="summary-card">
                    <h3>📊 Total Batches</h3>
                    <p style="font-size: 2em; margin: 0;">{total_batches}</p>
                </div>
                <div class="summary-card">
                    <h3>🎯 Total Demos</h3>
                    <p style="font-size: 2em; margin: 0;">{total_demos}</p>
                </div>
                <div class="summary-card">
                    <h3>✅ Successful Demos</h3>
                    <p style="font-size: 2em; margin: 0;">{total_successful}</p>
                </div>
                <div class="summary-card">
                    <h3>📈 Success Rate</h3>
                    <p style="font-size: 2em; margin: 0;">{overall_success_rate:.1%}</p>
                </div>
            </div>
    """
    
    # Individual batch results
    html_content += "<h2>📋 Batch Results</h2>"
    
    for batch in sorted(all_batches, key=lambda x: x['batch_completed'], reverse=True):
        batch_time = datetime.fromisoformat(batch['batch_completed']).strftime('%Y-%m-%d %H:%M')
        
        html_content += f"""
            <div class="batch-section">
                <h3>Batch {batch['batch_id']} - {batch_time}</h3>
                <p><strong>Success Rate:</strong> {batch['success_rate']:.1%} 
                   ({batch['successful_demos']}/{batch['total_demos']})</p>
                <p><strong>Total Time:</strong> {batch['total_execution_time']:.2f}s</p>
                
                <h4>Demo Results:</h4>
        """
        
        for result in batch['results']:
            status_class = "success" if result.get('success') else "failure"
            status_text = "✅ Success" if result.get('success') else "❌ Failed"
            
            html_content += f"""
                <div class="demo-result">
                    <strong>{result['demo_name']}</strong> - 
                    <span class="{status_class}">{status_text}</span> - 
                    {result.get('execution_time_seconds', 0):.2f}s
                </div>
            """
        
        html_content += "</div>"
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    # Save dashboard
    dashboard_file = f"{results_directory}/dashboard_{timestamp_str()}.html"
    with open(dashboard_file, 'w') as f:
        f.write(html_content)
    
    print(f"📊 Dashboard generated: {dashboard_file}")
    print(f"🌐 Open in browser: file://{os.path.abspath(dashboard_file)}")
    
    return dashboard_file
```

---

## 📖 Complete Demo Reference

### **Updated Main Demo File**

Here's how to enhance your `examples/demo_tasks.py` to include all the new functionality:

```python
"""
examples/demo_tasks.py - Enhanced Demo Collection

A comprehensive collection of demos showcasing the Playwright LangGraph Agent capabilities.
Run with: python examples/demo_tasks.py [--interactive|--benchmark|--batch]
"""

from config import load_env
load_env()  # <--- Load environment first

import os
import sys
import asyncio
from datetime import datetime
from agent.web_browsing_agent import WebBrowsingAgent
from toolkit.web_toolkit import export_json
from utils import print_banner, timestamp_str, ensure_dir

# Import all demo functions (add the new ones above)
# ... [All the demo functions defined above] ...

async def main():
    """Main demo execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Playwright LangGraph Agent Demos")
    parser.add_argument("--interactive", action="store_true", help="Run interactive demo selector")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--batch", choices=["standard", "advanced"], help="Run demo batch suite")
    parser.add_argument("--dashboard", action="store_true", help="Generate results dashboard")
    parser.add_argument("--demo", type=str, help="Run specific demo by name")
    
    args = parser.parse_args()
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Please set OPENAI_API_KEY environment variable")
        sys.exit(1)
    
    if args.interactive:
        interactive_demo_selector()
    elif args.benchmark:
        await run_performance_benchmark()
    elif args.batch == "standard":
        await run_standard_demo_suite()
    elif args.batch == "advanced":
        await run_advanced_demo_suite()
    elif args.dashboard:
        generate_demo_dashboard()
    elif args.demo:
        # Run specific demo
        demo_map = {
            "news": demo_news_extraction,
            "form": demo_form_filling,
            "simple": demo_simple_extraction,
            "links": demo_link_discovery,
            "search": demo_search_interaction,
            "ecommerce": demo_ecommerce_research,
            "aggregation": demo_news_aggregation,
            "competitive": demo_competitive_analysis,
            "docs": demo_documentation_mining
        }
        
        if args.demo in demo_map:
            await demo_map[args.demo]()
        else:
            print(f"❌ Unknown demo: {args.demo}")
            print(f"Available demos: {', '.join(demo_map.keys())}")
    else:
        # Default: run basic news extraction
        print_banner("🦜 Playwright LangGraph Agent Demo")
        print("Running default news extraction demo...")
        print("Use --help to see all available options")
        await demo_news_extraction()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 🎯 Summary & Key Takeaways

This comprehensive demo guide transforms your basic demo tasks into a **complete learning and testing system**:

### **📚 What You've Gained**
✅ **Progressive Learning Path**: From basic extraction to advanced workflows  
✅ **Real-World Examples**: E-commerce, news aggregation, competitive analysis  
✅ **Testing Infrastructure**: Automated benchmarking and validation  
✅ **Production Patterns**: Batch processing, error handling, monitoring  
✅ **Interactive Tools**: CLI selectors, dashboards, analysis tools  

### **🚀 How to Use This Guide**
1. **Start Simple**: Begin with `demo_simple_extraction()` to understand basics
2. **Learn Progressively**: Move through Level 1 → 2 → 3 → 4 demos
3. **Practice Exercises**: Complete the learning challenges 
4. **Test Everything**: Use the benchmark and testing tools
5. **Build Custom**: Adapt patterns for your specific use cases

### **🎖️ Best Practices Learned**
- **Clear Task Instructions**: Specific, actionable prompts work best
- **Proper Error Handling**: Always check results and handle failures gracefully  
- **Respectful Automation**: Include delays between requests
- **Comprehensive Logging**: Track everything for debugging and analysis
- **Modular Design**: Build reusable components and patterns

### **🔮 Next Steps**
- Customize demos for your specific industry or use case
- Integrate with your existing data pipelines
- Scale up to production workloads using batch processing
- Contribute new demo patterns back to the community

**You now have a complete toolkit for mastering autonomous web browsing agents!** 🎯🚀

---

*This guide represents the culmination of practical agent development knowledge. Use it as your foundation for building intelligent web automation solutions.*