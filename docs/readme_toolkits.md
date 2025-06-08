# 🧰 Web Toolkit - Complete Pedagogical Guide & Usage Examples

A **comprehensive, production-grade tutorial** for the `web_toolkit.py` module - the batch processing and data export backbone of your Playwright LangGraph Agent project. This guide provides deep understanding of batch automation patterns, data management strategies, and scalable web automation workflows.

---

## 📂 Architecture Context

```plaintext
playwright_langgraph_agent/
├── agent/
│   └── web_browsing_agent.py     # 🤖 The Brain (individual tasks)
├── browser/
│   └── playwright_manager.py     # 🌐 The Hands (browser actions)
└── toolkit/
    └── web_toolkit.py           # 🧰 This file - The Orchestrator
```

**Role in the System:**
- **Batch Orchestration**: Coordinate multiple agent tasks efficiently
- **Data Export**: Convert agent results into various formats (CSV, JSON, Excel)
- **Concurrency Management**: Control resource usage with intelligent parallelization
- **Result Aggregation**: Combine and analyze results from multiple sources
- **Production Scaling**: Enable enterprise-level web automation workflows

---

## 🎯 Design Philosophy & Core Principles

### **1. Separation of Scale**
```python
# ✅ Good: Toolkit handles scale, agent handles individual tasks
await run_batch(agent, url_tasks, max_concurrent=5)  # Toolkit manages many
result = await agent.execute_task(url, task, "extract")  # Agent handles one

# ❌ Avoid: Mixing batch logic with individual task logic
```

### **2. Resource-Aware Concurrency**
```python
# ✅ Good: Controlled concurrency prevents system overload
async def run_batch(agent, url_tasks, max_concurrent=3):
    semaphore = asyncio.Semaphore(max_concurrent)
    # Limits simultaneous browser instances

# ❌ Avoid: Uncontrolled parallelism that crashes systems
asyncio.gather(*[agent.execute_task(url, task) for url in urls])  # Could spawn 1000 browsers!
```

### **3. Format-Agnostic Export**
```python
# ✅ Good: Support multiple export formats
export_json(results, "data.json")      # For APIs and processing
export_csv(results, "data.csv")        # For Excel and analysis
export_xlsx(results, "data.xlsx")      # For business reporting

# ❌ Avoid: Hard-coding single export format
```

### **4. Fault-Tolerant Processing**
```python
# ✅ Good: Individual failures don't stop the entire batch
results = await run_batch_with_retry(agent, url_tasks)
successful = [r for r in results if r["success"]]
failed = [r for r in results if not r["success"]]

# ❌ Avoid: One failure stops everything
```

---

## 🚀 Complete API Reference with Examples

### **Core Batch Processing Functions**

#### **Basic Batch Execution**
```python
from toolkit.web_toolkit import run_batch, export_json, export_csv
from agent.web_browsing_agent import WebBrowsingAgent
import asyncio

async def basic_batch_example():
    """Run multiple web extraction tasks in parallel"""
    agent = WebBrowsingAgent(api_key="your-key", headless=True)
    
    # Define tasks
    url_tasks = [
        {
            "url": "https://news.ycombinator.com",
            "task": "Extract top 5 headlines",
            "task_type": "extract"
        },
        {
            "url": "https://techcrunch.com", 
            "task": "Find startup news headlines",
            "task_type": "extract"
        },
        {
            "url": "https://github.com/trending",
            "task": "Extract trending repository names and descriptions",
            "task_type": "extract"
        }
    ]
    
    # Run batch with controlled concurrency
    results = await run_batch(agent, url_tasks, max_concurrent=2)
    
    # Export results
    export_json(results, "news_batch_results.json")
    export_csv(results, "news_batch_results.csv")
    
    return results

# Run the example
results = asyncio.run(basic_batch_example())
print(f"Processed {len(results)} URLs")
```

#### **Advanced Batch Configuration**
```python
async def advanced_batch_example():
    """Demonstrate advanced batch processing with error handling"""
    agent = WebBrowsingAgent(api_key="your-key", headless=True)
    
    # Complex task configuration
    url_tasks = [
        {
            "url": "https://httpbin.org/forms/post",
            "task": "Fill out contact form",
            "task_type": "interact",
            "form_data": {
                "#name": "John Doe",
                "#email": "john@example.com",
                "#message": "Automated test message"
            }
        },
        {
            "url": "https://example.com",
            "task": "Search for specific content",
            "task_type": "search"
        },
        {
            "url": "https://invalid-url-that-will-fail.com",
            "task": "This will fail gracefully",
            "task_type": "extract"
        }
    ]
    
    # Run with higher concurrency for I/O bound tasks
    results = await run_batch(agent, url_tasks, max_concurrent=5)
    
    # Analyze results
    successful_results = [r for r in results if r["success"]]
    failed_results = [r for r in results if not r["success"]]
    
    print(f"✅ Successful: {len(successful_results)}")
    print(f"❌ Failed: {len(failed_results)}")
    
    # Export with metadata
    batch_summary = {
        "batch_timestamp": datetime.now().isoformat(),
        "total_tasks": len(url_tasks),
        "successful_tasks": len(successful_results),
        "failed_tasks": len(failed_results),
        "success_rate": len(successful_results) / len(url_tasks),
        "results": results
    }
    
    export_json(batch_summary, f"batch_summary_{timestamp_str()}.json")
    
    return batch_summary
```

---

### **Data Export Functions**

#### **JSON Export with Advanced Features**
```python
def export_json_advanced(results, filename="results.json", pretty=True, metadata=None):
    """Enhanced JSON export with metadata and formatting options"""
    from utils import timestamp_str
    
    # Add metadata if provided
    output_data = {
        "export_metadata": {
            "timestamp": timestamp_str(),
            "total_results": len(results),
            "exporter_version": "1.0",
            **(metadata or {})
        },
        "results": results
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        if pretty:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        else:
            json.dump(output_data, f, separators=(',', ':'))
    
    print(f"📄 Exported {len(results)} results to {filename}")

# Usage example
async def export_with_metadata_example():
    agent = WebBrowsingAgent(api_key="your-key", headless=True)
    
    url_tasks = [
        {"url": "https://example.com", "task": "Extract content", "task_type": "extract"}
    ]
    
    results = await run_batch(agent, url_tasks)
    
    # Export with custom metadata
    export_json_advanced(
        results, 
        "detailed_results.json",
        metadata={
            "project": "Web Research Project",
            "batch_id": "batch_001",
            "operator": "research_team"
        }
    )
```

#### **CSV Export with Nested Data Handling**
```python
def export_csv_enhanced(results, filename="results.csv", flatten_nested=True):
    """Enhanced CSV export with better nested data handling"""
    if not results:
        print("⚠️  No results to export")
        return
    
    def flatten_dict(d, parent_key='', sep='_'):
        """Recursively flatten nested dictionaries"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # Convert lists to string representation
                items.append((new_key, json.dumps(v)))
            else:
                items.append((new_key, v))
        return dict(items)
    
    # Process results
    processed_results = []
    for result in results:
        if flatten_nested:
            flattened = flatten_dict(result)
            processed_results.append(flattened)
        else:
            # Convert complex types to JSON strings
            processed = {}
            for k, v in result.items():
                if isinstance(v, (dict, list)):
                    processed[k] = json.dumps(v)
                else:
                    processed[k] = v
            processed_results.append(processed)
    
    # Get all possible column names
    all_columns = set()
    for result in processed_results:
        all_columns.update(result.keys())
    
    # Export to CSV
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=sorted(all_columns))
        writer.writeheader()
        writer.writerows(processed_results)
    
    print(f"📊 Exported {len(results)} results to {filename} with {len(all_columns)} columns")

# Usage example
async def csv_export_example():
    agent = WebBrowsingAgent(api_key="your-key", headless=True)
    
    # Run batch extraction
    url_tasks = [
        {"url": "https://news.ycombinator.com", "task": "Extract headlines", "task_type": "extract"},
        {"url": "https://github.com/trending", "task": "Extract repositories", "task_type": "extract"}
    ]
    
    results = await run_batch(agent, url_tasks)
    
    # Export with flattened nested data
    export_csv_enhanced(results, "flattened_results.csv", flatten_nested=True)
    
    # Export with JSON-encoded nested data
    export_csv_enhanced(results, "json_encoded_results.csv", flatten_nested=False)
```

#### **Excel Export with Multiple Sheets**
```python
def export_xlsx(results, filename="results.xlsx", sheet_name="Results"):
    """Export results to Excel format with enhanced formatting"""
    try:
        import pandas as pd
        from utils import timestamp_str
        
        # Convert results to DataFrame
        df = pd.json_normalize(results)
        
        # Create Excel writer with formatting
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Main results sheet
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Summary sheet
            summary_data = {
                'Metric': ['Total Results', 'Successful', 'Failed', 'Success Rate', 'Export Date'],
                'Value': [
                    len(results),
                    sum(1 for r in results if r.get('success', False)),
                    sum(1 for r in results if not r.get('success', False)),
                    f"{sum(1 for r in results if r.get('success', False)) / len(results):.1%}" if results else "0%",
                    timestamp_str("%Y-%m-%d %H:%M:%S")
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            # Format the sheets
            workbook = writer.book
            
            # Auto-adjust column widths
            for sheet_name in workbook.sheetnames:
                worksheet = workbook[sheet_name]
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        if cell.value:
                            max_length = max(max_length, len(str(cell.value)))
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column_letter].width = adjusted_width
        
        print(f"📈 Exported {len(results)} results to {filename}")
        
    except ImportError:
        print("⚠️  pandas and openpyxl required for Excel export")
        print("Install with: pip install pandas openpyxl")
        # Fallback to CSV
        export_csv_enhanced(results, filename.replace('.xlsx', '.csv'))

# Usage example
async def excel_export_example():
    agent = WebBrowsingAgent(api_key="your-key", headless=True)
    
    url_tasks = [
        {"url": "https://example.com", "task": "Extract data", "task_type": "extract"}
        for i in range(10)  # Generate sample tasks
    ]
    
    results = await run_batch(agent, url_tasks)
    export_xlsx(results, "comprehensive_results.xlsx", "Web_Extraction_Results")
```

---

### **Advanced Batch Processing Patterns**

#### **Batch Processing with Progress Tracking**
```python
async def run_batch_with_progress(agent, url_tasks, max_concurrent=3):
    """Run batch with live progress tracking"""
    try:
        from tqdm.asyncio import tqdm
        progress_bar = True
    except ImportError:
        progress_bar = False
        print("Install tqdm for progress bars: pip install tqdm")
    
    semaphore = asyncio.Semaphore(max_concurrent)
    completed_tasks = []
    
    async def run_one_with_progress(task, task_index):
        async with semaphore:
            try:
                result = await agent.execute_task(
                    url=task['url'],
                    task=task['task'],
                    task_type=task.get('task_type', 'extract'),
                    form_data=task.get('form_data')
                )
                # Add task metadata
                result['task_index'] = task_index
                result['original_task'] = task
                return result
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e),
                    'task_index': task_index,
                    'original_task': task,
                    'url': task.get('url', 'unknown')
                }
    
    # Create tasks with indices
    indexed_tasks = [(task, i) for i, task in enumerate(url_tasks)]
    
    if progress_bar:
        # Use tqdm for progress tracking
        tasks = [run_one_with_progress(task, idx) for task, idx in indexed_tasks]
        results = []
        for coro in tqdm.as_completed(tasks, desc="Processing URLs"):
            result = await coro
            results.append(result)
            completed_tasks.append(result)
            
            # Print status updates
            success_count = sum(1 for r in completed_tasks if r.get('success'))
            total_completed = len(completed_tasks)
            print(f"\r✅ {success_count}/{total_completed} successful", end='', flush=True)
    else:
        # Simple completion tracking
        results = []
        for i, (task, idx) in enumerate(indexed_tasks):
            result = await run_one_with_progress(task, idx)
            results.append(result)
            print(f"Completed {i+1}/{len(url_tasks)}: {task['url']}")
    
    # Sort results by original task index
    results.sort(key=lambda x: x.get('task_index', 0))
    
    return results

# Usage example
async def progress_tracking_example():
    agent = WebBrowsingAgent(api_key="your-key", headless=True)
    
    # Generate many tasks for demonstration
    url_tasks = [
        {
            "url": f"https://httpbin.org/delay/{i%3}",  # Simulate variable load times
            "task": f"Extract data from page {i}",
            "task_type": "extract"
        }
        for i in range(20)
    ]
    
    print("🚀 Starting batch with progress tracking...")
    results = await run_batch_with_progress(agent, url_tasks, max_concurrent=4)
    
    # Summary
    successful = sum(1 for r in results if r.get('success'))
    print(f"\n📊 Batch Complete: {successful}/{len(results)} successful")
    
    return results
```

#### **Retry Logic and Error Recovery**
```python
async def run_batch_with_retry(agent, url_tasks, max_concurrent=3, max_retries=2):
    """Run batch with automatic retry on failures"""
    import asyncio
    from utils import timestamp_str
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def run_with_retry(task, task_id):
        async with semaphore:
            last_error = None
            
            for attempt in range(max_retries + 1):
                try:
                    result = await agent.execute_task(
                        url=task['url'],
                        task=task['task'],
                        task_type=task.get('task_type', 'extract'),
                        form_data=task.get('form_data')
                    )
                    
                    # Add retry metadata
                    result['retry_attempts'] = attempt
                    result['task_id'] = task_id
                    
                    if result.get('success'):
                        if attempt > 0:
                            print(f"✅ {task['url']} succeeded on attempt {attempt + 1}")
                        return result
                    else:
                        last_error = result.get('error', 'Unknown error')
                        if attempt < max_retries:
                            wait_time = 2 ** attempt  # Exponential backoff
                            print(f"⏳ Retrying {task['url']} in {wait_time}s (attempt {attempt + 1}/{max_retries + 1})")
                            await asyncio.sleep(wait_time)
                
                except Exception as e:
                    last_error = str(e)
                    if attempt < max_retries:
                        wait_time = 2 ** attempt
                        print(f"⏳ Exception for {task['url']}, retrying in {wait_time}s: {e}")
                        await asyncio.sleep(wait_time)
            
            # All retries exhausted
            return {
                'success': False,
                'error': f"Failed after {max_retries + 1} attempts: {last_error}",
                'retry_attempts': max_retries + 1,
                'task_id': task_id,
                'url': task.get('url', 'unknown'),
                'timestamp': timestamp_str()
            }
    
    # Execute with retry logic
    tasks = [run_with_retry(task, i) for i, task in enumerate(url_tasks)]
    results = await asyncio.gather(*tasks)
    
    return results

# Usage example
async def retry_example():
    agent = WebBrowsingAgent(api_key="your-key", headless=True)
    
    # Mix of reliable and unreliable URLs
    url_tasks = [
        {"url": "https://httpbin.org/status/200", "task": "Should work", "task_type": "extract"},
        {"url": "https://httpbin.org/status/500", "task": "Will fail", "task_type": "extract"},
        {"url": "https://definitely-invalid-url.com", "task": "Will fail", "task_type": "extract"},
        {"url": "https://example.com", "task": "Should work", "task_type": "extract"}
    ]
    
    results = await run_batch_with_retry(agent, url_tasks, max_retries=2)
    
    # Analyze retry patterns
    retry_stats = {}
    for result in results:
        attempts = result.get('retry_attempts', 0)
        retry_stats[attempts] = retry_stats.get(attempts, 0) + 1
    
    print("📈 Retry Statistics:")
    for attempts, count in sorted(retry_stats.items()):
        print(f"  {attempts} attempts: {count} tasks")
    
    return results
```

#### **Batch Processing with Rate Limiting**
```python
async def run_batch_with_rate_limit(agent, url_tasks, requests_per_second=1, max_concurrent=3):
    """Run batch with rate limiting to respect website policies"""
    import asyncio
    import time
    
    semaphore = asyncio.Semaphore(max_concurrent)
    rate_limiter = asyncio.Semaphore(requests_per_second)
    request_times = []
    
    async def rate_limited_task(task, task_id):
        async with rate_limiter:
            # Track request timing
            current_time = time.time()
            request_times.append(current_time)
            
            # Clean old request times (older than 1 second)
            cutoff_time = current_time - 1.0
            request_times[:] = [t for t in request_times if t > cutoff_time]
            
            # If we're at the rate limit, wait
            if len(request_times) >= requests_per_second:
                sleep_time = 1.0 - (current_time - request_times[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
            
            async with semaphore:
                return await agent.execute_task(
                    url=task['url'],
                    task=task['task'],
                    task_type=task.get('task_type', 'extract'),
                    form_data=task.get('form_data')
                )
    
    # Execute with rate limiting
    tasks = [rate_limited_task(task, i) for i, task in enumerate(url_tasks)]
    results = await asyncio.gather(*tasks)
    
    return results

# Usage example
async def rate_limited_example():
    agent = WebBrowsingAgent(api_key="your-key", headless=True)
    
    # Batch of requests to the same domain (be respectful!)
    url_tasks = [
        {"url": f"https://httpbin.org/delay/1", "task": f"Request {i}", "task_type": "extract"}
        for i in range(10)
    ]
    
    print("🚀 Starting rate-limited batch (1 request/second)...")
    start_time = time.time()
    
    results = await run_batch_with_rate_limit(
        agent, 
        url_tasks, 
        requests_per_second=1,  # Very conservative
        max_concurrent=2
    )
    
    elapsed_time = time.time() - start_time
    print(f"⏱️  Completed {len(results)} requests in {elapsed_time:.1f} seconds")
    print(f"📊 Average rate: {len(results)/elapsed_time:.2f} requests/second")
    
    return results
```

---

### **Data Processing and Analysis Utilities**

#### **Result Aggregation and Analysis**
```python
def analyze_batch_results(results):
    """Comprehensive analysis of batch processing results"""
    from collections import Counter, defaultdict
    from utils import timestamp_str
    
    analysis = {
        "timestamp": timestamp_str(),
        "total_tasks": len(results),
        "successful_tasks": 0,
        "failed_tasks": 0,
        "success_rate": 0.0,
        "error_analysis": {},
        "performance_metrics": {},
        "data_quality": {}
    }
    
    # Basic success/failure analysis
    successful_results = [r for r in results if r.get('success', False)]
    failed_results = [r for r in results if not r.get('success', False)]
    
    analysis["successful_tasks"] = len(successful_results)
    analysis["failed_tasks"] = len(failed_results)
    analysis["success_rate"] = len(successful_results) / len(results) if results else 0
    
    # Error analysis
    error_types = Counter()
    error_urls = defaultdict(list)
    
    for result in failed_results:
        error = result.get('error', 'Unknown error')
        error_types[error] += 1
        error_urls[error].append(result.get('url', 'Unknown URL'))
    
    analysis["error_analysis"] = {
        "error_types": dict(error_types),
        "most_common_errors": error_types.most_common(5),
        "error_urls": dict(error_urls)
    }
    
    # Performance metrics
    processing_times = []
    extracted_data_sizes = []
    
    for result in successful_results:
        # Extract timing info if available
        nav_history = result.get('navigation_history', [])
        if nav_history:
            processing_times.append(len(nav_history))  # Rough proxy for processing time
        
        # Extract data size info
        extracted_data = result.get('extracted_data', {})
        if isinstance(extracted_data, dict):
            elements = extracted_data.get('elements', [])
            extracted_data_sizes.append(len(elements))
    
    if processing_times:
        analysis["performance_metrics"] = {
            "avg_processing_steps": sum(processing_times) / len(processing_times),
            "min_processing_steps": min(processing_times),
            "max_processing_steps": max(processing_times)
        }
    
    if extracted_data_sizes:
        analysis["data_quality"] = {
            "avg_elements_extracted": sum(extracted_data_sizes) / len(extracted_data_sizes),
            "min_elements_extracted": min(extracted_data_sizes),
            "max_elements_extracted": max(extracted_data_sizes),
            "total_elements_extracted": sum(extracted_data_sizes)
        }
    
    # URL domain analysis
    domain_stats = Counter()
    for result in results:
        url = result.get('url', '')
        if url:
            try:
                from urllib.parse import urlparse
                domain = urlparse(url).netloc
                domain_stats[domain] += 1
            except:
                domain_stats['invalid_url'] += 1
    
    analysis["domain_analysis"] = {
        "domains_processed": dict(domain_stats),
        "unique_domains": len(domain_stats),
        "most_processed_domains": domain_stats.most_common(10)
    }
    
    return analysis

# Usage example
async def analysis_example():
    agent = WebBrowsingAgent(api_key="your-key", headless=True)
    
    # Run a varied batch for analysis
    url_tasks = [
        {"url": "https://news.ycombinator.com", "task": "Extract headlines", "task_type": "extract"},
        {"url": "https://github.com/trending", "task": "Extract repos", "task_type": "extract"},
        {"url": "https://invalid-url.com", "task": "This will fail", "task_type": "extract"},
        {"url": "https://example.com", "task": "Extract content", "task_type": "extract"},
        {"url": "https://httpbin.org/html", "task": "Extract HTML", "task_type": "extract"}
    ]
    
    results = await run_batch(agent, url_tasks)
    analysis = analyze_batch_results(results)
    
    # Export analysis
    export_json(analysis, f"batch_analysis_{timestamp_str()}.json")
    
    # Print summary
    print("📊 Batch Analysis Summary:")
    print(f"✅ Success Rate: {analysis['success_rate']:.1%}")
    print(f"🌐 Domains Processed: {analysis['domain_analysis']['unique_domains']}")
    print(f"📈 Avg Elements Extracted: {analysis['data_quality'].get('avg_elements_extracted', 0):.1f}")
    
    if analysis["error_analysis"]["most_common_errors"]:
        print("🔍 Most Common Errors:")
        for error, count in analysis["error_analysis"]["most_common_errors"][:3]:
            print(f"  • {error}: {count} times")
    
    return analysis
```

#### **Data Filtering and Transformation**
```python
def filter_and_transform_results(results, filters=None, transformations=None):
    """Apply filters and transformations to batch results"""
    
    # Default filters
    default_filters = {
        'success_only': lambda r: r.get('success', False),
        'has_data': lambda r: bool(r.get('extracted_data')),
        'min_elements': lambda r, min_count=1: len(r.get('extracted_data', {}).get('elements', [])) >= min_count
    }
    
    # Default transformations
    default_transformations = {
        'add_timestamp': lambda r: {**r, 'processed_at': timestamp_str()},
        'clean_urls': lambda r: {**r, 'url': r.get('url', '').strip().lower()},
        'extract_domain': lambda r: {
            **r, 
            'domain': urlparse(r.get('url', '')).netloc if r.get('url') else 'unknown'
        }
    }
    
    # Apply filters
    filtered_results = results
    if filters:
        for filter_name, filter_args in filters.items():
            if filter_name in default_filters:
                filter_func = default_filters[filter_name]
                if isinstance(filter_args, dict):
                    filtered_results = [r for r in filtered_results if filter_func(r, **filter_args)]
                else:
                    filtered_results = [r for r in filtered_results if filter_func(r)]
    
    # Apply transformations
    transformed_results = filtered_results
    if transformations:
        for transform_name in transformations:
            if transform_name in default_transformations:
                transform_func = default_transformations[transform_name]
                transformed_results = [transform_func(r) for r in transformed_results]
    
    return transformed_results

# Usage example
async def filtering_example():
    agent = WebBrowsingAgent(api_key="your-key", headless=True)
    
    url_tasks = [
        {"url": "https://example.com", "task": "Extract content", "task_type": "extract"},
        {"url": "https://invalid-url.com", "task": "Will fail", "task_type": "extract"},
        {"url": "https://httpbin.org/html", "task": "Extract HTML", "task_type": "extract"}
    ]
    
    results = await run_batch(agent, url_tasks)
    
    # Filter and transform
    processed_results = filter_and_transform_results(
        results,
        filters={
            'success_only': True,
            'min_elements': {'min_count': 2}
        },
        transformations=['add_timestamp', 'extract_domain']
    )
    
    print(f"📊 Original results: {len(results)}")
    print(f"🔍 Filtered results: {len(processed_results)}")
    
    return processed_results
```

---

### **Specialized Batch Processing Patterns**

#### **Domain-Grouped Batch Processing**
```python
async def run_batch_by_domain(agent, url_tasks, max_concurrent_per_domain=1):
    """Process URLs grouped by domain with per-domain concurrency limits"""
    from collections import defaultdict
    from urllib.parse import urlparse
    
    # Group tasks by domain
    domain_groups = defaultdict(list)
    for i, task in enumerate(url_tasks):
        try:
            domain = urlparse(task['url']).netloc
            domain_groups[domain].append((i, task))
        except:
            domain_groups['invalid'].append((i, task))
    
    print(f"🌐 Processing {len(domain_groups)} domains with {len(url_tasks)} total tasks")
    
    # Create semaphores for each domain
    domain_semaphores = {
        domain: asyncio.Semaphore(max_concurrent_per_domain) 
        for domain in domain_groups.keys()
    }
    
    async def process_domain_task(domain, task_index, task):
        async with domain_semaphores[domain]:
            # Add delay between requests to same domain
            await asyncio.sleep(1)  # Be respectful to servers
            
            result = await agent.execute_task(
                url=task['url'],
                task=task['task'],
                task_type=task.get('task_type', 'extract'),
                form_data=task.get('form_data')
            )
            
            result['domain'] = domain
            result['original_index'] = task_index
            return result
    
    # Create all tasks
    all_tasks = []
    for domain, domain_tasks in domain_groups.items():
        for task_index, task in domain_tasks:
            all_tasks.append(process_domain_task(domain, task_index, task))
    
    # Execute all tasks
    results = await asyncio.gather(*all_tasks)
    
    # Sort results back to original order
    results.sort(key=lambda x: x.get('original_index', 0))
    
    # Print domain statistics
    domain_stats = defaultdict(lambda: {'total': 0, 'successful': 0})
    for result in results:
        domain = result.get('domain', 'unknown')
        domain_stats[domain]['total'] += 1
        if result.get('success'):
            domain_stats[domain]['successful'] += 1
    
    print("\n📈 Domain Processing Statistics:")
    for domain, stats in domain_stats.items():
        success_rate = stats['successful'] / stats['total'] if stats['total'] > 0 else 0
        print(f"  {domain}: {stats['successful']}/{stats['total']} ({success_rate:.1%})")
    
    return results

# Usage example
async def domain_grouped_example():
    agent = WebBrowsingAgent(api_key="your-key", headless=True)
    
    # URLs from different domains
    url_tasks = [
        {"url": "https://example.com/page1", "task": "Extract content", "task_type": "extract"},
        {"url": "https://example.com/page2", "task": "Extract content", "task_type": "extract"},
        {"url": "https://httpbin.org/html", "task": "Extract HTML", "task_type": "extract"},
        {"url": "https://httpbin.org/json", "task": "Extract JSON", "task_type": "extract"},
        {"url": "https://github.com/trending", "task": "Extract repos", "task_type": "extract"},
        {"url": "https://news.ycombinator.com", "task": "Extract headlines", "task_type": "extract"}
    ]
    
    results = await run_batch_by_domain(agent, url_tasks, max_concurrent_per_domain=1)
    export_json(results, "domain_grouped_results.json")
    
    return results
```

#### **Streaming Batch Processing for Large Datasets**
```python
async def run_streaming_batch(agent, url_generator, batch_size=10, max_concurrent=3):
    """Process URLs in streaming batches for memory efficiency with large datasets"""
    
    async def process_batch_chunk(batch_chunk):
        """Process a single batch chunk"""
        return await run_batch(agent, batch_chunk, max_concurrent)
    
    def chunk_generator(iterable, chunk_size):
        """Generate chunks from an iterable"""
        chunk = []
        for item in iterable:
            chunk.append(item)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk
    
    all_results = []
    processed_count = 0
    
    print(f"🚀 Starting streaming batch processing (batch_size={batch_size})")
    
    # Process in chunks
    for batch_num, batch_chunk in enumerate(chunk_generator(url_generator, batch_size)):
        print(f"\n📦 Processing batch {batch_num + 1} ({len(batch_chunk)} tasks)...")
        
        batch_results = await process_batch_chunk(batch_chunk)
        all_results.extend(batch_results)
        processed_count += len(batch_chunk)
        
        # Progress update
        successful_in_batch = sum(1 for r in batch_results if r.get('success'))
        print(f"✅ Batch {batch_num + 1} complete: {successful_in_batch}/{len(batch_chunk)} successful")
        
        # Save intermediate results (optional)
        export_json(batch_results, f"intermediate_batch_{batch_num + 1:03d}.json")
        
        # Brief pause between batches
        await asyncio.sleep(2)
    
    print(f"\n🎉 Streaming batch processing complete: {processed_count} total tasks processed")
    return all_results

# Usage example with generator
async def streaming_example():
    agent = WebBrowsingAgent(api_key="your-key", headless=True)
    
    # Generator function for large dataset
    def url_task_generator():
        base_urls = [
            "https://httpbin.org/delay/1",
            "https://example.com",
            "https://httpbin.org/html"
        ]
        
        for i in range(50):  # Generate 50 tasks
            url = base_urls[i % len(base_urls)]
            yield {
                "url": f"{url}?task_id={i}",
                "task": f"Extract content from task {i}",
                "task_type": "extract"
            }
    
    # Process in streaming batches
    results = await run_streaming_batch(
        agent, 
        url_task_generator(), 
        batch_size=5, 
        max_concurrent=2
    )
    
    # Final export
    export_json(results, "streaming_final_results.json")
    
    print(f"📊 Final results: {len(results)} tasks processed")
    
    return results
```

---

### **Enterprise-Grade Features**

#### **Batch Processing with Checkpointing**
```python
import pickle
import os

class BatchCheckpoint:
    """Handle batch processing checkpoints for crash recovery"""
    
    def __init__(self, checkpoint_file="batch_checkpoint.pkl"):
        self.checkpoint_file = checkpoint_file
        self.completed_indices = set()
        self.results = []
        self.load_checkpoint()
    
    def load_checkpoint(self):
        """Load existing checkpoint if available"""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'rb') as f:
                    checkpoint_data = pickle.load(f)
                    self.completed_indices = checkpoint_data.get('completed_indices', set())
                    self.results = checkpoint_data.get('results', [])
                print(f"📂 Loaded checkpoint: {len(self.completed_indices)} completed tasks")
            except Exception as e:
                print(f"⚠️  Failed to load checkpoint: {e}")
    
    def save_checkpoint(self):
        """Save current progress to checkpoint file"""
        try:
            checkpoint_data = {
                'completed_indices': self.completed_indices,
                'results': self.results,
                'timestamp': timestamp_str()
            }
            with open(self.checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
        except Exception as e:
            print(f"⚠️  Failed to save checkpoint: {e}")
    
    def is_completed(self, task_index):
        """Check if task is already completed"""
        return task_index in self.completed_indices
    
    def mark_completed(self, task_index, result):
        """Mark task as completed and save result"""
        self.completed_indices.add(task_index)
        self.results.append(result)
        
        # Save checkpoint every 10 completed tasks
        if len(self.completed_indices) % 10 == 0:
            self.save_checkpoint()

async def run_batch_with_checkpoint(agent, url_tasks, max_concurrent=3, checkpoint_file=None):
    """Run batch with automatic checkpointing for crash recovery"""
    
    checkpoint = BatchCheckpoint(checkpoint_file or f"checkpoint_{timestamp_str()}.pkl")
    
    # Filter out already completed tasks
    remaining_tasks = [
        (i, task) for i, task in enumerate(url_tasks) 
        if not checkpoint.is_completed(i)
    ]
    
    if len(remaining_tasks) < len(url_tasks):
        completed_count = len(url_tasks) - len(remaining_tasks)
        print(f"🔄 Resuming batch: {completed_count} tasks already completed, {len(remaining_tasks)} remaining")
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_checkpoint(task_index, task):
        async with semaphore:
            try:
                result = await agent.execute_task(
                    url=task['url'],
                    task=task['task'],
                    task_type=task.get('task_type', 'extract'),
                    form_data=task.get('form_data')
                )
                result['task_index'] = task_index
                checkpoint.mark_completed(task_index, result)
                print(f"✅ Completed task {task_index}: {task['url']}")
                return result
                
            except Exception as e:
                error_result = {
                    'success': False,
                    'error': str(e),
                    'task_index': task_index,
                    'url': task.get('url', 'unknown')
                }
                checkpoint.mark_completed(task_index, error_result)
                print(f"❌ Failed task {task_index}: {e}")
                return error_result
    
    # Process remaining tasks
    if remaining_tasks:
        tasks = [process_with_checkpoint(idx, task) for idx, task in remaining_tasks]
        await asyncio.gather(*tasks)
    
    # Final checkpoint save
    checkpoint.save_checkpoint()
    
    # Sort results by task index
    all_results = sorted(checkpoint.results, key=lambda x: x.get('task_index', 0))
    
    print(f"🎉 Batch processing complete with checkpointing: {len(all_results)} tasks")
    
    return all_results

# Usage example
async def checkpoint_example():
    agent = WebBrowsingAgent(api_key="your-key", headless=True)
    
    # Large batch that might be interrupted
    url_tasks = [
        {"url": f"https://httpbin.org/delay/{i%3}", "task": f"Task {i}", "task_type": "extract"}
        for i in range(30)
    ]
    
    # Run with checkpointing
    results = await run_batch_with_checkpoint(
        agent, 
        url_tasks, 
        max_concurrent=3,
        checkpoint_file="large_batch_checkpoint.pkl"
    )
    
    export_json(results, "checkpointed_results.json")
    
    return results
```

#### **Multi-Agent Batch Processing**
```python
class AgentPool:
    """Manage multiple agent instances for high-throughput processing"""
    
    def __init__(self, api_key, pool_size=3, headless=True):
        self.api_key = api_key
        self.pool_size = pool_size
        self.headless = headless
        self.agents = []
        self.agent_semaphore = asyncio.Semaphore(pool_size)
    
    async def initialize_pool(self):
        """Initialize the agent pool"""
        print(f"🤖 Initializing agent pool with {self.pool_size} agents...")
        
        for i in range(self.pool_size):
            agent = WebBrowsingAgent(self.api_key, headless=self.headless)
            self.agents.append(agent)
        
        print(f"✅ Agent pool ready: {len(self.agents)} agents available")
    
    async def get_agent(self):
        """Get an available agent from the pool"""
        await self.agent_semaphore.acquire()
        return self.agents.pop()
    
    def return_agent(self, agent):
        """Return agent to the pool"""
        self.agents.append(agent)
        self.agent_semaphore.release()
    
    async def cleanup_pool(self):
        """Cleanup all agents in the pool"""
        for agent in self.agents:
            try:
                await agent.browser.cleanup()
            except:
                pass

async def run_multi_agent_batch(url_tasks, api_key, agent_pool_size=3, max_concurrent=5):
    """Run batch processing with multiple agent instances"""
    
    pool = AgentPool(api_key, pool_size=agent_pool_size, headless=True)
    await pool.initialize_pool()
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_agent_pool(task, task_index):
        async with semaphore:
            agent = await pool.get_agent()
            try:
                result = await agent.execute_task(
                    url=task['url'],
                    task=task['task'],
                    task_type=task.get('task_type', 'extract'),
                    form_data=task.get('form_data')
                )
                result['task_index'] = task_index
                result['agent_id'] = id(agent)  # Track which agent processed this
                return result
            finally:
                pool.return_agent(agent)
    
    try:
        # Execute all tasks
        tasks = [process_with_agent_pool(task, i) for i, task in enumerate(url_tasks)]
        results = await asyncio.gather(*tasks)
        
        # Sort by original task order
        results.sort(key=lambda x: x.get('task_index', 0))
        
        return results
        
    finally:
        await pool.cleanup_pool()

# Usage example
async def multi_agent_example():
    # Large batch requiring multiple agents
    url_tasks = [
        {"url": f"https://httpbin.org/delay/{i%3}", "task": f"Extract task {i}", "task_type": "extract"}
        for i in range(50)
    ]
    
    print(f"🚀 Starting multi-agent batch processing for {len(url_tasks)} tasks...")
    start_time = time.time()
    
    results = await run_multi_agent_batch(
        url_tasks, 
        api_key="your-key",
        agent_pool_size=4,  # 4 concurrent agents
        max_concurrent=8    # 8 total concurrent tasks
    )
    
    elapsed_time = time.time() - start_time
    successful = sum(1 for r in results if r.get('success'))
    
    print(f"⏱️  Multi-agent processing complete:")
    print(f"   📊 Results: {successful}/{len(results)} successful")
    print(f"   ⏱️  Time: {elapsed_time:.1f} seconds")
    print(f"   📈 Rate: {len(results)/elapsed_time:.2f} tasks/second")
    
    # Analyze agent usage
    agent_usage = {}
    for result in results:
        agent_id = result.get('agent_id', 'unknown')
        agent_usage[agent_id] = agent_usage.get(agent_id, 0) + 1
    
    print(f"🤖 Agent utilization:")
    for agent_id, task_count in agent_usage.items():
        print(f"   Agent {agent_id}: {task_count} tasks")
    
    export_json(results, "multi_agent_results.json")
    
    return results
```

---

### **Integration Patterns & Real-World Examples**

#### **E-commerce Price Monitoring**
```python
async def ecommerce_price_monitoring():
    """Monitor product prices across multiple e-commerce sites"""
    from utils import ensure_dir, timestamp_str
    
    agent = WebBrowsingAgent(api_key="your-key", headless=True)
    
    # Product monitoring configuration
    products_to_monitor = [
        {
            "url": "https://example-store.com/product/laptop-123",
            "task": "Extract product name, price, and availability",
            "task_type": "extract",
            "product_id": "laptop-123",
            "category": "electronics"
        },
        {
            "url": "https://another-store.com/smartphones/phone-456",
            "task": "Find current price and stock status",
            "task_type": "extract", 
            "product_id": "phone-456",
            "category": "electronics"
        }
        # Add more products...
    ]
    
    print(f"🛒 Starting price monitoring for {len(products_to_monitor)} products...")
    
    # Run monitoring batch
    results = await run_batch_with_retry(
        agent, 
        products_to_monitor, 
        max_concurrent=2,  # Be respectful to e-commerce sites
        max_retries=2
    )
    
    # Process and structure price data
    price_data = []
    for result in results:
        if result.get('success'):
            extracted_data = result.get('extracted_data', {})
            
            price_record = {
                "timestamp": timestamp_str(),
                "product_id": result.get('product_id'),
                "category": result.get('category'),
                "url": result.get('url'),
                "extracted_info": extracted_data,
                "monitoring_successful": True
            }
        else:
            price_record = {
                "timestamp": timestamp_str(),
                "product_id": result.get('product_id'),
                "category": result.get('category'), 
                "url": result.get('url'),
                "error": result.get('error'),
                "monitoring_successful": False
            }
        
        price_data.append(price_record)
    
    # Save results with timestamp
    ensure_dir("price_monitoring")
    export_json(price_data, f"price_monitoring/prices_{timestamp_str()}.json")
    export_csv_enhanced(price_data, f"price_monitoring/prices_{timestamp_str()}.csv")
    
    # Generate monitoring report
    successful_monitoring = sum(1 for p in price_data if p['monitoring_successful'])
    
    print(f"📊 Price monitoring complete:")
    print(f"   ✅ Successfully monitored: {successful_monitoring}/{len(price_data)} products")
    print(f"   📁 Results saved to price_monitoring/")
    
    return price_data
```

#### **Content Research Pipeline**
```python
async def content_research_pipeline(research_topics, max_sources_per_topic=5):
    """Research multiple topics across various sources"""
    from itertools import product
    
    agent = WebBrowsingAgent(api_key="your-key", headless=True)
    
    # Source websites for research
    research_sources = [
        {"base_url": "https://news.ycombinator.com", "type": "tech_news"},
        {"base_url": "https://techcrunch.com", "type": "startup_news"},
        {"base_url": "https://github.com/trending", "type": "code_trends"}
    ]
    
    # Generate research tasks
    research_tasks = []
    for topic in research_topics:
        for source in research_sources[:max_sources_per_topic]:
            task = {
                "url": source["base_url"],
                "task": f"Search for and extract information about '{topic}'",
                "task_type": "search",
                "research_topic": topic,
                "source_type": source["type"],
                "source_url": source["base_url"]
            }
            research_tasks.append(task)
    
    print(f"🔬 Starting research pipeline for {len(research_topics)} topics across {len(research_sources)} sources...")
    print(f"📋 Total research tasks: {len(research_tasks)}")
    
    # Execute research batch with domain grouping
    results = await run_batch_by_domain(
        agent, 
        research_tasks, 
        max_concurrent_per_domain=1  # Respectful crawling
    )
    
    # Organize results by topic
    research_results = {}
    for result in results:
        topic = result.get('research_topic', 'unknown')
        if topic not in research_results:
            research_results[topic] = {
                'topic': topic,
                'sources_researched': 0,
                'successful_sources': 0,
                'findings': [],
                'errors': []
            }
        
        research_results[topic]['sources_researched'] += 1
        
        if result.get('success'):
            research_results[topic]['successful_sources'] += 1
            research_results[topic]['findings'].append({
                'source_type': result.get('source_type'),
                'source_url': result.get('source_url'),
                'extracted_data': result.get('extracted_data'),
                'timestamp': result.get('timestamp')
            })
        else:
            research_results[topic]['errors'].append({
                'source_type': result.get('source_type'),
                'source_url': result.get('source_url'),
                'error': result.get('error')
            })
    
    # Generate research report
    ensure_dir("research_reports")
    timestamp = timestamp_str()
    
    # Save detailed results
    export_json(research_results, f"research_reports/research_{timestamp}.json")
    
    # Create summary report
    summary_report = {
        "research_timestamp": timestamp,
        "topics_researched": len(research_topics),
        "total_sources_checked": len(research_tasks),
        "topic_summaries": {}
    }
    
    for topic, data in research_results.items():
        summary_report["topic_summaries"][topic] = {
            "sources_checked": data['sources_researched'],
            "successful_sources": data['successful_sources'],
            "success_rate": data['successful_sources'] / data['sources_researched'] if data['sources_researched'] > 0 else 0,
            "findings_count": len(data['findings'])
        }
    
    export_json(summary_report, f"research_reports/summary_{timestamp}.json")
    
    print(f"📊 Research pipeline complete:")
    for topic, summary in summary_report["topic_summaries"].items():
        print(f"   📝 {topic}: {summary['successful_sources']}/{summary['sources_checked']} sources ({summary['success_rate']:.1%})")
    
    return research_results

# Usage example
async def research_example():
    research_topics = [
        "artificial intelligence trends 2024",
        "web automation tools",
        "python async programming",
        "browser automation best practices"
    ]
    
    results = await content_research_pipeline(research_topics, max_sources_per_topic=3)
    
    return results
```

#### **Competitive Analysis Automation**
```python
async def competitive_analysis_batch(competitor_urls, analysis_categories):
    """Automated competitive analysis across multiple competitors and categories"""
    
    agent = WebBrowsingAgent(api_key="your-key", headless=True)
    
    # Generate analysis tasks for each competitor and category
    analysis_tasks = []
    for competitor_url in competitor_urls:
        for category in analysis_categories:
            task = {
                "url": competitor_url,
                "task": f"Analyze {category} on this website",
                "task_type": "extract",
                "competitor": competitor_url,
                "analysis_category": category
            }
            analysis_tasks.append(task)
    
    print(f"🏢 Starting competitive analysis:")
    print(f"   🏭 Competitors: {len(competitor_urls)}")
    print(f"   📊 Categories: {len(analysis_categories)}")
    print(f"   📋 Total tasks: {len(analysis_tasks)}")
    
    # Execute analysis with progress tracking
    results = await run_batch_with_progress(
        agent, 
        analysis_tasks, 
        max_concurrent=2
    )
    
    # Organize results by competitor and category
    competitive_data = {}
    for result in results:
        competitor = result.get('competitor', 'unknown')
        category = result.get('analysis_category', 'unknown')
        
        if competitor not in competitive_data:
            competitive_data[competitor] = {}
        
        competitive_data[competitor][category] = {
            'success': result.get('success', False),
            'extracted_data': result.get('extracted_data', {}),
            'error': result.get('error'),
            'timestamp': result.get('timestamp')
        }
    
    # Generate competitive analysis report
    ensure_dir("competitive_analysis")
    timestamp = timestamp_str()
    
    # Create comparison matrix
    comparison_matrix = []
    for category in analysis_categories:
        row = {'category': category}
        for competitor in competitor_urls:
            competitor_name = urlparse(competitor).netloc
            success = competitive_data.get(competitor, {}).get(category, {}).get('success', False)
            row[competitor_name] = 'Success' if success else 'Failed'
        comparison_matrix.append(row)
    
    # Export results
    export_json(competitive_data, f"competitive_analysis/detailed_{timestamp}.json")
    export_csv_enhanced(comparison_matrix, f"competitive_analysis/matrix_{timestamp}.csv")
    
    # Generate executive summary
    summary = {
        "analysis_date": timestamp,
        "competitors_analyzed": len(competitor_urls),
        "categories_analyzed": len(analysis_categories),
        "total_data_points": len(analysis_tasks),
        "success_by_competitor": {},
        "success_by_category": {}
    }
    
    # Calculate success rates
    for competitor in competitor_urls:
        competitor_name = urlparse(competitor).netloc
        competitor_data = competitive_data.get(competitor, {})
        successful = sum(1 for cat_data in competitor_data.values() if cat_data.get('success'))
        total = len(competitor_data)
        summary["success_by_competitor"][competitor_name] = {
            "successful": successful,
            "total": total,
            "success_rate": successful / total if total > 0 else 0
        }
    
    for category in analysis_categories:
        successful = 0
        total = 0
        for competitor_data in competitive_data.values():
            if category in competitor_data:
                total += 1
                if competitor_data[category].get('success'):
                    successful += 1
        summary["success_by_category"][category] = {
            "successful": successful,
            "total": total,
            "success_rate": successful / total if total > 0 else 0
        }
    
    export_json(summary, f"competitive_analysis/summary_{timestamp}.json")
    
    print(f"📊 Competitive analysis complete:")
    print(f"   📁 Results saved to competitive_analysis/")
    print(f"   📈 Overall data points collected: {sum(summary['success_by_competitor'].values(), key=lambda x: x['successful'])}")
    
    return competitive_data, summary

# Usage example
async def competitive_analysis_example():
    competitors = [
        "https://competitor1.com",
        "https://competitor2.com", 
        "https://competitor3.com"
    ]
    
    analysis_categories = [
        "pricing information",
        "key features and benefits",
        "contact information and support",
        "technology stack indicators",
        "customer testimonials"
    ]
    
    detailed_results, summary = await competitive_analysis_batch(competitors, analysis_categories)
    
    return detailed_results, summary
```

---

### **Testing & Quality Assurance**

#### **Comprehensive Test Suite for Batch Processing**
```python
import pytest
import tempfile
import os

class TestWebToolkit:
    """Comprehensive test suite for web toolkit functionality"""
    
    @pytest.fixture
    def sample_results(self):
        """Generate sample results for testing"""
        return [
            {
                "success": True,
                "url": "https://example.com",
                "extracted_data": {
                    "title": "Example Domain",
                    "elements": [
                        {"tag": "h1", "text": "Example Domain"},
                        {"tag": "p", "text": "This domain is for use in illustrative examples."}
                    ]
                },
                "timestamp": "2024-12-08T14:30:52",
                "navigation_history": ["Navigated to example.com", "Extracted data"]
            },
            {
                "success": False,
                "url": "https://invalid-url.com",
                "error": "Navigation failed: DNS resolution failed",
                "timestamp": "2024-12-08T14:31:15",
                "navigation_history": ["Failed to navigate"]
            }
        ]
    
    @pytest.fixture
    def temp_directory(self):
        """Create temporary directory for test files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    def test_export_json(self, sample_results, temp_directory):
        """Test JSON export functionality"""
        from toolkit.web_toolkit import export_json
        
        json_file = os.path.join(temp_directory, "test_results.json")
        export_json(sample_results, json_file)
        
        # Verify file exists and contains correct data
        assert os.path.exists(json_file)
        
        with open(json_file, 'r') as f:
            import json
            loaded_data = json.load(f)
            assert len(loaded_data) == len(sample_results)
            assert loaded_data[0]["success"] == True
            assert loaded_data[1]["success"] == False
    
    def test_export_csv(self, sample_results, temp_directory):
        """Test CSV export functionality"""
        from toolkit.web_toolkit import export_csv
        
        csv_file = os.path.join(temp_directory, "test_results.csv")
        export_csv(sample_results, csv_file)
        
        # Verify file exists and has correct structure
        assert os.path.exists(csv_file)
        
        import csv
        with open(csv_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == len(sample_results)
            
            # Check that complex data is JSON-encoded
            first_row = rows[0]
            assert 'extracted_data' in first_row
            assert 'navigation_history' in first_row
    
    def test_export_enhanced_csv(self, sample_results, temp_directory):
        """Test enhanced CSV export with flattening"""
        csv_file = os.path.join(temp_directory, "test_enhanced.csv")
        export_csv_enhanced(sample_results, csv_file, flatten_nested=True)
        
        assert os.path.exists(csv_file)
        
        import csv
        with open(csv_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            # Check that nested data is flattened
            first_row = rows[0]
            assert any('extracted_data_' in key for key in first_row.keys())
    
    @pytest.mark.asyncio
    async def test_run_batch_basic(self):
        """Test basic batch processing functionality"""
        from unittest.mock import AsyncMock, MagicMock
        
        # Mock agent
        mock_agent = AsyncMock()
        mock_agent.execute_task.return_value = {
            "success": True,
            "url": "https://example.com",
            "extracted_data": {"test": "data"}
        }
        
        # Test tasks
        url_tasks = [
            {"url": "https://example.com", "task": "test", "task_type": "extract"},
            {"url": "https://example2.com", "task": "test2", "task_type": "extract"}
        ]
        
        # Run batch
        from toolkit.web_toolkit import run_batch
        results = await run_batch(mock_agent, url_tasks, max_concurrent=2)
        
        # Verify results
        assert len(results) == 2
        assert all(r["success"] for r in results)
        assert mock_agent.execute_task.call_count == 2
    
    @pytest.mark.asyncio
    async def test_run_batch_with_retry_logic(self):
        """Test batch processing with retry functionality"""
        from unittest.mock import AsyncMock
        
        # Mock agent that fails first, succeeds second
        mock_agent = AsyncMock()
        mock_agent.execute_task.side_effect = [
            {"success": False, "error": "Temporary failure"},
            {"success": True, "url": "https://example.com", "extracted_data": {"test": "data"}}
        ]
        
        url_tasks = [{"url": "https://example.com", "task": "test", "task_type": "extract"}]
        
        # This would test the retry logic (implementation depends on your retry function)
        results = await run_batch_with_retry(mock_agent, url_tasks, max_retries=1)
        
        assert len(results) == 1
        # The retry logic should have tried twice
        assert mock_agent.execute_task.call_count == 2
    
    def test_analyze_batch_results(self, sample_results):
        """Test batch results analysis functionality"""
        analysis = analyze_batch_results(sample_results)
        
        # Check analysis structure
        assert "total_tasks" in analysis
        assert "successful_tasks" in analysis
        assert "failed_tasks" in analysis
        assert "success_rate" in analysis
        assert "error_analysis" in analysis
        
        # Check values
        assert analysis["total_tasks"] == 2
        assert analysis["successful_tasks"] == 1
        assert analysis["failed_tasks"] == 1
        assert analysis["success_rate"] == 0.5
    
    def test_filter_and_transform_results(self, sample_results):
        """Test result filtering and transformation"""
        # Filter for successful results only
        filtered = filter_and_transform_results(
            sample_results,
            filters={'success_only': True},
            transformations=['add_timestamp']
        )
        
        # Should only have 1 successful result
        assert len(filtered) == 1
        assert filtered[0]["success"] == True
        assert "processed_at" in filtered[0]  # Added by transformation

# Integration tests with real agent (requires API key)
class TestWebToolkitIntegration:
    """Integration tests that require actual agent execution"""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_real_batch_processing(self):
        """Test batch processing with real agent (requires OPENAI_API_KEY)"""
        import os
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set - skipping integration test")
        
        from agent.web_browsing_agent import WebBrowsingAgent
        from toolkit.web_toolkit import run_batch
        
        agent = WebBrowsingAgent(api_key, headless=True)
        
        # Simple, reliable test tasks
        url_tasks = [
            {"url": "https://httpbin.org/html", "task": "Extract HTML content", "task_type": "extract"}
        ]
        
        results = await run_batch(agent, url_tasks, max_concurrent=1)
        
        assert len(results) == 1
        assert "success" in results[0]
        assert "extracted_data" in results[0]
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_domain_grouped_processing(self):
        """Test domain-grouped batch processing"""
        import os
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set - skipping integration test")
        
        from agent.web_browsing_agent import WebBrowsingAgent
        
        agent = WebBrowsingAgent(api_key, headless=True)
        
        url_tasks = [
            {"url": "https://httpbin.org/html", "task": "Extract content", "task_type": "extract"},
            {"url": "https://httpbin.org/json", "task": "Extract JSON", "task_type": "extract"},
            {"url": "https://example.com", "task": "Extract content", "task_type": "extract"}
        ]
        
        results = await run_batch_by_domain(agent, url_tasks, max_concurrent_per_domain=1)
        
        assert len(results) == 3
        # Check that domain information is added
        assert all("domain" in result for result in results)

# Performance benchmarks
class TestWebToolkitPerformance:
    """Performance tests for batch processing"""
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_batch_concurrency_scaling(self):
        """Test how batch processing scales with different concurrency levels"""
        from unittest.mock import AsyncMock
        import time
        
        # Mock agent with simulated delay
        async def mock_execute_task(*args, **kwargs):
            await asyncio.sleep(0.1)  # Simulate 100ms processing time
            return {"success": True, "url": "test", "extracted_data": {}}
        
        mock_agent = AsyncMock()
        mock_agent.execute_task = mock_execute_task
        
        # Test with 10 tasks
        url_tasks = [
            {"url": f"https://test{i}.com", "task": "test", "task_type": "extract"}
            for i in range(10)
        ]
        
        # Test different concurrency levels
        concurrency_levels = [1, 3, 5]
        performance_results = {}
        
        for concurrency in concurrency_levels:
            start_time = time.time()
            results = await run_batch(mock_agent, url_tasks, max_concurrent=concurrency)
            elapsed_time = time.time() - start_time
            
            performance_results[concurrency] = {
                "elapsed_time": elapsed_time,
                "tasks_per_second": len(results) / elapsed_time
            }
        
        # Higher concurrency should be faster (up to a point)
        assert performance_results[3]["elapsed_time"] < performance_results[1]["elapsed_time"]
        
        print("Performance results:")
        for concurrency, metrics in performance_results.items():
            print(f"  Concurrency {concurrency}: {metrics['elapsed_time']:.2f}s, {metrics['tasks_per_second']:.2f} tasks/sec")

# Example test runner
async def run_toolkit_tests():
    """Run toolkit tests manually (for development)"""
    print("🧪 Running Web Toolkit Tests...")
    
    # Basic functionality tests
    test_instance = TestWebToolkit()
    
    # Generate sample data
    sample_results = [
        {"success": True, "url": "https://example.com", "data": {"key": "value"}},
        {"success": False, "url": "https://failed.com", "error": "Connection failed"}
    ]
    
    # Test exports
    with tempfile.TemporaryDirectory() as temp_dir:
        print("📄 Testing JSON export...")
        test_instance.test_export_json(sample_results, temp_dir)
        
        print("📊 Testing CSV export...")
        test_instance.test_export_csv(sample_results, temp_dir)
        
        print("📈 Testing enhanced CSV export...")
        test_instance.test_export_enhanced_csv(sample_results, temp_dir)
    
    # Test analysis
    print("🔍 Testing result analysis...")
    test_instance.test_analyze_batch_results(sample_results)
    
    # Test filtering
    print("🔧 Testing filtering and transformation...")
    test_instance.test_filter_and_transform_results(sample_results)
    
    print("✅ All toolkit tests passed!")

# Run tests if executed directly
if __name__ == "__main__":
    asyncio.run(run_toolkit_tests())
```

---

### **Performance Optimization & Best Practices**

#### **Memory-Efficient Batch Processing**
```python
import psutil
import gc

class MemoryEfficientBatchProcessor:
    """Batch processor optimized for memory usage with large datasets"""
    
    def __init__(self, agent, memory_limit_mb=1024):
        self.agent = agent
        self.memory_limit_bytes = memory_limit_mb * 1024 * 1024
        self.processed_count = 0
        self.results_buffer = []
        self.buffer_size = 100  # Process in chunks of 100
    
    def get_memory_usage(self):
        """Get current memory usage in bytes"""
        process = psutil.Process()
        return process.memory_info().rss
    
    async def process_with_memory_management(self, url_tasks, export_callback=None):
        """Process tasks with automatic memory management"""
        
        total_tasks = len(url_tasks)
        all_results = []
        
        for i in range(0, total_tasks, self.buffer_size):
            chunk = url_tasks[i:i + self.buffer_size]
            chunk_start = i + 1
            chunk_end = min(i + self.buffer_size, total_tasks)
            
            print(f"🔄 Processing chunk {chunk_start}-{chunk_end}/{total_tasks}")
            
            # Monitor memory before processing
            memory_before = self.get_memory_usage()
            
            # Process chunk
            chunk_results = await run_batch(self.agent, chunk, max_concurrent=3)
            all_results.extend(chunk_results)
            
            # Monitor memory after processing
            memory_after = self.get_memory_usage()
            memory_increase = memory_after - memory_before
            
            print(f"📊 Memory usage: {memory_after / 1024 / 1024:.1f} MB (+{memory_increase / 1024 / 1024:.1f} MB)")
            
            # Export intermediate results if callback provided
            if export_callback:
                export_callback(chunk_results, f"chunk_{i // self.buffer_size + 1:03d}")
            
            # Force garbage collection if memory is getting high
            if memory_after > self.memory_limit_bytes:
                print("🧹 Memory limit reached, forcing garbage collection...")
                gc.collect()
                
                memory_after_gc = self.get_memory_usage()
                print(f"📉 Memory after GC: {memory_after_gc / 1024 / 1024:.1f} MB")
            
            # Brief pause between chunks
            await asyncio.sleep(1)
        
        return all_results

# Usage example
async def memory_efficient_example():
    """Example of memory-efficient batch processing"""
    agent = WebBrowsingAgent(api_key="your-key", headless=True)
    
    # Generate large task set
    url_tasks = [
        {"url": f"https://httpbin.org/delay/{i%3}", "task": f"Task {i}", "task_type": "extract"}
        for i in range(1000)  # Large batch
    ]
    
    # Create processor with 512MB memory limit
    processor = MemoryEfficientBatchProcessor(agent, memory_limit_mb=512)
    
    def chunk_export_callback(chunk_results, chunk_name):
        """Export each chunk as it's processed"""
        ensure_dir("memory_efficient_results")
        export_json(chunk_results, f"memory_efficient_results/{chunk_name}.json")
    
    print(f"🚀 Starting memory-efficient processing of {len(url_tasks)} tasks...")
    
    results = await processor.process_with_memory_management(
        url_tasks, 
        export_callback=chunk_export_callback
    )
    
    print(f"✅ Memory-efficient processing complete: {len(results)} results")
    
    return results
```

#### **Performance Monitoring and Optimization**
```python
import time
import statistics
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class BatchMetrics:
    """Comprehensive metrics for batch processing performance"""
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    task_times: List[float] = None
    memory_usage: List[float] = None
    concurrency_level: int = 0
    
    def __post_init__(self):
        if self.task_times is None:
            self.task_times = []
        if self.memory_usage is None:
            self.memory_usage = []
    
    @property
    def total_time(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def success_rate(self) -> float:
        return self.successful_tasks / self.total_tasks if self.total_tasks > 0 else 0
    
    @property
    def average_task_time(self) -> float:
        return statistics.mean(self.task_times) if self.task_times else 0
    
    @property
    def tasks_per_second(self) -> float:
        return self.total_tasks / self.total_time if self.total_time > 0 else 0

class PerformanceMonitor:
    """Monitor and optimize batch processing performance"""
    
    def __init__(self):
        self.metrics_history: List[BatchMetrics] = []
    
    async def run_monitored_batch(self, agent, url_tasks, max_concurrent=3):
        """Run batch with comprehensive performance monitoring"""
        
        metrics = BatchMetrics(
            total_tasks=len(url_tasks),
            concurrency_level=max_concurrent,
            start_time=time.time()
        )
        
        # Enhanced batch function with timing
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def timed_task(task, task_index):
            task_start = time.time()
            async with semaphore:
                try:
                    result = await agent.execute_task(
                        url=task['url'],
                        task=task['task'],
                        task_type=task.get('task_type', 'extract'),
                        form_data=task.get('form_data')
                    )
                    
                    task_end = time.time()
                    task_time = task_end - task_start
                    metrics.task_times.append(task_time)
                    
                    if result.get('success'):
                        metrics.successful_tasks += 1
                    else:
                        metrics.failed_tasks += 1
                    
                    # Monitor memory usage periodically
                    if task_index % 10 == 0:
                        try:
                            import psutil
                            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
                            metrics.memory_usage.append(memory_mb)
                        except ImportError:
                            pass
                    
                    return result
                    
                except Exception as e:
                    metrics.failed_tasks += 1
                    task_end = time.time()
                    metrics.task_times.append(task_end - task_start)
                    return {
                        'success': False,
                        'error': str(e),
                        'url': task.get('url', 'unknown')
                    }
        
        # Execute tasks
        tasks = [timed_task(task, i) for i, task in enumerate(url_tasks)]
        results = await asyncio.gather(*tasks)
        
        metrics.end_time = time.time()
        self.metrics_history.append(metrics)
        
        # Print performance summary
        self.print_performance_summary(metrics)
        
        return results, metrics
    
    def print_performance_summary(self, metrics: BatchMetrics):
        """Print comprehensive performance summary"""
        print("\n📈 Performance Summary:")
        print(f"   ⏱️  Total Time: {metrics.total_time:.2f} seconds")
        print(f"   📊 Success Rate: {metrics.success_rate:.1%} ({metrics.successful_tasks}/{metrics.total_tasks})")
        print(f"   🚀 Tasks/Second: {metrics.tasks_per_second:.2f}")
        print(f"   ⚡ Avg Task Time: {metrics.average_task_time:.2f}s")
        print(f"   🔄 Concurrency: {metrics.concurrency_level}")
        
        if metrics.task_times:
            print(f"   📏 Min Task Time: {min(metrics.task_times):.2f}s")
            print(f"   📏 Max Task Time: {max(metrics.task_times):.2f}s")
            print(f"   📏 Task Time StdDev: {statistics.stdev(metrics.task_times):.2f}s")
        
        if metrics.memory_usage:
            print(f"   💾 Avg Memory: {statistics.mean(metrics.memory_usage):.1f} MB")
            print(f"   💾 Peak Memory: {max(metrics.memory_usage):.1f} MB")
    
    def optimize_concurrency(self, agent, url_tasks, test_concurrency_levels=None):
        """Find optimal concurrency level through testing"""
        if test_concurrency_levels is None:
            test_concurrency_levels = [1, 2, 3, 5, 8]
        
        # Use small subset for testing
        test_tasks = url_tasks[:min(10, len(url_tasks))]
        optimization_results = {}
        
        print(f"🔬 Testing concurrency levels: {test_concurrency_levels}")
        print(f"📋 Using {len(test_tasks)} tasks for optimization")
        
        for concurrency in test_concurrency_levels:
            print(f"\n🧪 Testing concurrency level: {concurrency}")
            
            start_time = time.time()
            results, metrics = await self.run_monitored_batch(
                agent, test_tasks, max_concurrent=concurrency
            )
            
            optimization_results[concurrency] = {
                'tasks_per_second': metrics.tasks_per_second,
                'success_rate': metrics.success_rate,
                'avg_task_time': metrics.average_task_time,
                'total_time': metrics.total_time
            }
        
        # Find optimal concurrency (best tasks/second with >90% success rate)
        viable_options = {
            c: metrics for c, metrics in optimization_results.items()
            if metrics['success_rate'] >= 0.9
        }
        
        if viable_options:
            optimal_concurrency = max(
                viable_options.keys(),
                key=lambda c: viable_options[c]['tasks_per_second']
            )
        else:
            optimal_concurrency = min(optimization_results.keys())
        
        print(f"\n🎯 Optimization Results:")
        print(f"   🏆 Optimal Concurrency: {optimal_concurrency}")
        print(f"   📈 Expected Performance: {optimization_results[optimal_concurrency]['tasks_per_second']:.2f} tasks/sec")
        
        return optimal_concurrency, optimization_results

# Usage example
async def performance_optimization_example():
    """Example of performance monitoring and optimization"""
    agent = WebBrowsingAgent(api_key="your-key", headless=True)
    monitor = PerformanceMonitor()
    
    # Test tasks
    url_tasks = [
        {"url": f"https://httpbin.org/delay/{i%3}", "task": f"Task {i}", "task_type": "extract"}
        for i in range(20)
    ]
    
    # Find optimal concurrency
    optimal_concurrency, optimization_results = await monitor.optimize_concurrency(
        agent, url_tasks, test_concurrency_levels=[1, 2, 4, 6]
    )
    
    # Run full batch with optimal settings
    print(f"\n🚀 Running full batch with optimal concurrency: {optimal_concurrency}")
    results, final_metrics = await monitor.run_monitored_batch(
        agent, url_tasks, max_concurrent=optimal_concurrency
    )
    
    # Export performance data
    performance_report = {
        "optimization_results": optimization_results,
        "optimal_concurrency": optimal_concurrency,
        "final_metrics": {
            "total_tasks": final_metrics.total_tasks,
            "success_rate": final_metrics.success_rate,
            "tasks_per_second": final_metrics.tasks_per_second,
            "total_time": final_metrics.total_time,
            "average_task_time": final_metrics.average_task_time
        },
        "timestamp": timestamp_str()
    }
    
    export_json(performance_report, f"performance_report_{timestamp_str()}.json")
    
    return results, final_metrics
```

---

### **Production Deployment & Monitoring**

#### **Production-Ready Batch Processor**
```python
import logging
import signal
import sys
from pathlib import Path

class ProductionBatchProcessor:
    """Production-ready batch processor with enterprise features"""
    
    def __init__(self, api_key, config=None):
        self.api_key = api_key
        self.config = config or self.get_default_config()
        self.agent = None
        self.results = []
        self.processed_count = 0
        self.failed_count = 0
        self.start_time = None
        self.shutdown_requested = False
        
        # Setup logging
        self.setup_production_logging()
        self.logger = logging.getLogger(__name__)
        
        # Setup signal handlers for graceful shutdown
        self.setup_signal_handlers()
    
    def get_default_config(self):
        """Default production configuration"""
        return {
            "max_concurrent": 5,
            "max_retries": 3,
            "retry_delay": 2.0,
            "checkpoint_interval": 50,
            "memory_limit_mb": 1024,
            "export_format": "json",
            "enable_monitoring": True,
            "enable_checkpointing": True,
            "graceful_shutdown_timeout": 30
        }
    
    def setup_production_logging(self):
        """Setup comprehensive logging for production"""
        ensure_dir("logs")
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Setup file handlers
        file_handler = logging.FileHandler(f'logs/batch_processor_{timestamp_str()}.log')
        file_handler.setFormatter(detailed_formatter)
        file_handler.setLevel(logging.DEBUG)
        
        # Setup console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(simple_formatter)
        console_handler.setLevel(logging.INFO)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            self.logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
            self.shutdown_requested = True
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def initialize(self):
        """Initialize the batch processor"""
        self.logger.info("Initializing production batch processor...")
        
        try:
            self.agent = WebBrowsingAgent(self.api_key, headless=True)
            self.logger.info("Agent initialized successfully")
            
            # Create output directories
            ensure_dir("batch_output")
            ensure_dir("batch_checkpoints")
            ensure_dir("batch_monitoring")
            
            self.logger.info("Production batch processor ready")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize batch processor: {e}")
            raise
    
    async def process_batch_production(self, url_tasks, batch_id=None):
        """Process batch with full production features"""
        
        if batch_id is None:
            batch_id = f"batch_{timestamp_str()}"
        
        self.start_time = time.time()
        self.logger.info(f"Starting production batch {batch_id} with {len(url_tasks)} tasks")
        
        # Initialize checkpoint if enabled
        checkpoint = None
        if self.config["enable_checkpointing"]:
            checkpoint = BatchCheckpoint(f"batch_checkpoints/{batch_id}.pkl")
        
        # Initialize monitoring
        if self.config["enable_monitoring"]:
            monitor = PerformanceMonitor()
        
        try:
            # Process with all production features
            if checkpoint:
                results = await self.process_with_checkpoint(url_tasks, checkpoint, batch_id)
            else:
                results = await run_batch_with_retry(
                    self.agent, 
                    url_tasks, 
                    max_concurrent=self.config["max_concurrent"],
                    max_retries=self.config["max_retries"]
                )
            
            self.results = results
            self.processed_count = len(results)
            self.failed_count = sum(1 for r in results if not r.get('success', False))
            
            # Export results
            await self.export_production_results(results, batch_id)
            
            # Generate monitoring report
            if self.config["enable_monitoring"]:
                await self.generate_monitoring_report(results, batch_id)
            
            # Final logging
            elapsed_time = time.time() - self.start_time
            success_rate = (self.processed_count - self.failed_count) / self.processed_count
            
            self.logger.info(f"Batch {batch_id} completed successfully")
            self.logger.info(f"Processed: {self.processed_count}, Failed: {self.failed_count}")
            self.logger.info(f"Success rate: {success_rate:.1%}, Time: {elapsed_time:.2f}s")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch {batch_id} failed: {e}")
            
            # Save partial results if any
            if self.results:
                await self.export_production_results(self.results, f"{batch_id}_partial")
            
            raise
        
        finally:
            await self.cleanup()
    
    async def process_with_checkpoint(self, url_tasks, checkpoint, batch_id):
        """Process with checkpointing support"""
        
        remaining_tasks = [
            (i, task) for i, task in enumerate(url_tasks)
            if not checkpoint.is_completed(i)
        ]
        
        self.logger.info(f"Checkpoint recovery: {len(remaining_tasks)} tasks remaining")
        
        semaphore = asyncio.Semaphore(self.config["max_concurrent"])
        
        async def process_with_checkpoint_task(task_index, task):
            if self.shutdown_requested:
                self.logger.warning("Shutdown requested, skipping task")
                return None
            
            async with semaphore:
                for attempt in range(self.config["max_retries"] + 1):
                    try:
                        result = await self.agent.execute_task(
                            url=task['url'],
                            task=task['task'],
                            task_type=task.get('task_type', 'extract'),
                            form_data=task.get('form_data')
                        )
                        
                        result['task_index'] = task_index
                        result['batch_id'] = batch_id
                        checkpoint.mark_completed(task_index, result)
                        
                        self.logger.debug(f"Completed task {task_index}: {task['url']}")
                        return result
                        
                    except Exception as e:
                        if attempt < self.config["max_retries"]:
                            wait_time = self.config["retry_delay"] * (2 ** attempt)
                            self.logger.warning(f"Task {task_index} failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                            await asyncio.sleep(wait_time)
                        else:
                            error_result = {
                                'success': False,
                                'error': str(e),
                                'task_index': task_index,
                                'batch_id': batch_id,
                                'url': task.get('url', 'unknown')
                            }
                            checkpoint.mark_completed(task_index, error_result)
                            self.logger.error(f"Task {task_index} failed after {self.config['max_retries']} retries: {e}")
                            return error_result
        
        # Process remaining tasks
        if remaining_tasks:
            tasks = [process_with_checkpoint_task(idx, task) for idx, task in remaining_tasks]
            await asyncio.gather(*tasks)
        
        # Sort and return all results
        all_results = sorted(checkpoint.results, key=lambda x: x.get('task_index', 0))
        return all_results
    
    async def export_production_results(self, results, batch_id):
        """Export results with production-grade features"""
        timestamp = timestamp_str()
        
        # Prepare metadata
        metadata = {
            "batch_id": batch_id,
            "export_timestamp": timestamp,
            "total_results": len(results),
            "successful_results": sum(1 for r in results if r.get('success', False)),
            "failed_results": sum(1 for r in results if not r.get('success', False)),
            "processing_time": time.time() - self.start_time if self.start_time else 0,
            "config_used": self.config
        }
        
        # Export in multiple formats
        base_filename = f"batch_output/{batch_id}_{timestamp}"
        
        # JSON with metadata
        export_data = {
            "metadata": metadata,
            "results": results
        }
        export_json(export_data, f"{base_filename}.json")
        
        # CSV for analysis
        export_csv_enhanced(results, f"{base_filename}.csv", flatten_nested=True)
        
        # Excel for business reporting
        try:
            export_xlsx(results, f"{base_filename}.xlsx", f"{batch_id}_Results")
        except Exception as e:
            self.logger.warning(f"Excel export failed: {e}")
        
        # Summary report
        summary_report = {
            "batch_summary": metadata,
            "performance_metrics": {
                "success_rate": metadata["successful_results"] / metadata["total_results"],
                "tasks_per_second": metadata["total_results"] / metadata["processing_time"] if metadata["processing_time"] > 0 else 0,
                "average_processing_time": metadata["processing_time"] / metadata["total_results"] if metadata["total_results"] > 0 else 0
            }
        }
        export_json(summary_report, f"{base_filename}_summary.json")
        
        self.logger.info(f"Results exported to {base_filename}.*")
    
    async def generate_monitoring_report(self, results, batch_id):
        """Generate comprehensive monitoring report"""
        
        # Analyze results
        analysis = analyze_batch_results(results)
        
        # System metrics
        try:
            import psutil
            system_metrics = {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage_percent": psutil.disk_usage('/').percent
            }
        except ImportError:
            system_metrics = {"note": "psutil not available for system metrics"}
        
        # Monitoring report
        monitoring_report = {
            "batch_id": batch_id,
            "timestamp": timestamp_str(),
            "batch_analysis": analysis,
            "system_metrics": system_metrics,
            "configuration": self.config
        }
        
        export_json(monitoring_report, f"batch_monitoring/{batch_id}_monitoring.json")
        self.logger.info(f"Monitoring report saved for batch {batch_id}")
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.agent and hasattr(self.agent, 'browser'):
            try:
                await self.agent.browser.cleanup()
            except Exception as e:
                self.logger.warning(f"Browser cleanup failed: {e}")
        
        self.logger.info("Cleanup completed")

# Usage example for production deployment
async def production_deployment_example():
    """Example of production deployment"""
    
    # Production configuration
    config = {
        "max_concurrent": 8,
        "max_retries": 3,
        "retry_delay": 2.0,
        "checkpoint_interval": 25,
        "memory_limit_mb": 2048,
        "export_format": "all",
        "enable_monitoring": True,
        "enable_checkpointing": True
    }
    
    # Initialize processor
    processor = ProductionBatchProcessor(
        api_key="your-production-api-key",
        config=config
    )
    
    await processor.initialize()
    
    # Large production batch
    url_tasks = [
        {"url": f"https://api.example.com/data/{i}", "task": f"Extract data {i}", "task_type": "extract"}
        for i in range(500)  # Large production batch
    ]
    
    try:
        results = await processor.process_batch_production(url_tasks, "production_batch_001")
        
        print(f"✅ Production batch completed successfully: {len(results)} results")
        
    except KeyboardInterrupt:
        print("🛑 Batch processing interrupted by user")
    except Exception as e:
        print(f"❌ Production batch failed: {e}")
    
    return results
```

---

## 🎓 Learning Exercises & Best Practices

### **Exercise 1: Build a News Aggregator**
```python
async def exercise_news_aggregator():
    """
    Exercise: Build a comprehensive news aggregation system
    
    Requirements:
    1. Monitor 10+ news sources
    2. Extract headlines, dates, and categories
    3. Implement rate limiting (max 1 req/sec per domain)
    4. Export results in JSON and CSV
    5. Generate analytics report
    
    Your task: Complete this implementation
    """
    agent = WebBrowsingAgent(api_key="your-key", headless=True)
    
    news_sources = [
        {"url": "https://news.ycombinator.com", "category": "tech"},
        {"url": "https://techcrunch.com", "category": "startup"},
        {"url": "https://arstechnica.com", "category": "tech"},
        # Add 7+ more sources
    ]
    
    # TODO: Implement the news aggregator
    # Hints:
    # - Use run_batch_by_domain for respectful crawling
    # - Implement custom export with categorization
    # - Add duplicate detection
    # - Generate trending topics report
    
    pass  # Replace with your implementation

# Solution template provided separately
```

### **Exercise 2: E-commerce Price Tracker**
```python
async def exercise_price_tracker():
    """
    Exercise: Build a price tracking system
    
    Requirements:
    1. Track prices across multiple e-commerce sites
    2. Detect price changes over time
    3. Support multiple product categories
    4. Send alerts for significant price drops
    5. Generate price history reports
    
    Your task: Implement a complete price tracking solution
    """
    
    # Product definitions
    products = [
        {
            "name": "Laptop Model X",
            "urls": [
                "https://store1.com/laptop-x",
                "https://store2.com/laptop-x",
                "https://store3.com/laptop-x"
            ],
            "target_price": 999.99,
            "alert_threshold": 0.1  # 10% price drop
        }
        # Add more products
    ]
    
    # TODO: Implement price tracking system
    # Hints:
    # - Store historical price data
    # - Implement price change detection
    # - Add email/notification system
    # - Create price trend visualization
    
    pass  # Replace with your implementation
```

### **Exercise 3: Content Research Assistant**
```python
async def exercise_research_assistant():
    """
    Exercise: Build an intelligent research assistant
    
    Requirements:
    1. Research topics across academic, news, and industry sources
    2. Extract and summarize key findings
    3. Identify trending topics and themes
    4. Generate comprehensive research reports
    5. Support citation tracking and verification
    
    Your task: Create a research automation system
    """
    
    research_query = {
        "topic": "Machine Learning in Healthcare",
        "sources": ["academic", "news", "industry"],
        "depth": "comprehensive",
        "time_range": "last_12_months"
    }
    
    # TODO: Implement research assistant
    # Hints:
    # - Define source-specific extraction strategies
    # - Implement content similarity detection
    # - Add automatic summarization
    # - Generate citation-ready reports
    
    pass  # Replace with your implementation
```

---

## 🔧 Advanced Configuration Examples

### **Custom Export Formats**
```python
def export_xml(results, filename="results.xml"):
    """Export results to XML format"""
    import xml.etree.ElementTree as ET
    
    root = ET.Element("batch_results")
    root.set("total_count", str(len(results)))
    root.set("export_timestamp", timestamp_str())
    
    for i, result in enumerate(results):
        result_elem = ET.SubElement(root, "result")
        result_elem.set("index", str(i))
        
        for key, value in result.items():
            elem = ET.SubElement(result_elem, key)
            if isinstance(value, (dict, list)):
                elem.text = json.dumps(value)
            else:
                elem.text = str(value)
    
    tree = ET.ElementTree(root)
    tree.write(filename, encoding='utf-8', xml_declaration=True)
    print(f"📄 Exported {len(results)} results to {filename}")

def export_markdown_report(results, filename="report.md"):
    """Export results as a formatted Markdown report"""
    
    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"# Batch Processing Report\n\n")
        f.write(f"**Generated:** {timestamp_str()}\n\n")
        f.write(f"## Summary\n\n")
        f.write(f"- **Total Tasks:** {len(results)}\n")
        f.write(f"- **Successful:** {len(successful)} ({len(successful)/len(results):.1%})\n")
        f.write(f"- **Failed:** {len(failed)} ({len(failed)/len(results):.1%})\n\n")
        
        if successful:
            f.write(f"## Successful Results\n\n")
            for i, result in enumerate(successful[:10], 1):  # Show top 10
                f.write(f"### {i}. {result.get('url', 'Unknown URL')}\n\n")
                extracted = result.get('extracted_data', {})
                if extracted:
                    elements = extracted.get('elements', [])
                    f.write(f"- **Elements Found:** {len(elements)}\n")
                    if elements:
                        f.write(f"- **Sample Element:** {elements[0].get('text', '')[:100]}...\n")
                f.write(f"\n")
        
        if failed:
            f.write(f"## Failed Results\n\n")
            for i, result in enumerate(failed[:5], 1):  # Show top 5 failures
                f.write(f"### {i}. {result.get('url', 'Unknown URL')}\n\n")
                f.write(f"- **Error:** {result.get('error', 'Unknown error')}\n\n")
    
    print(f"📝 Exported report to {filename}")
```

### **Custom Batch Strategies**
```python
class PriorityBatchProcessor:
    """Process tasks based on priority levels"""
    
    def __init__(self, agent):
        self.agent = agent
    
    async def run_priority_batch(self, prioritized_tasks, max_concurrent=3):
        """Process tasks in priority order with concurrency control"""
        
        # Sort tasks by priority (higher numbers = higher priority)
        sorted_tasks = sorted(
            prioritized_tasks, 
            key=lambda x: x.get('priority', 0), 
            reverse=True
        )
        
        # Group by priority levels
        priority_groups = {}
        for task in sorted_tasks:
            priority = task.get('priority', 0)
            if priority not in priority_groups:
                priority_groups[priority] = []
            priority_groups[priority].append(task)
        
        all_results = []
        
        # Process each priority group sequentially
        for priority in sorted(priority_groups.keys(), reverse=True):
            tasks = priority_groups[priority]
            print(f"🔥 Processing priority {priority}: {len(tasks)} tasks")
            
            group_results = await run_batch(self.agent, tasks, max_concurrent)
            all_results.extend(group_results)
            
            # Brief pause between priority groups
            await asyncio.sleep(1)
        
        return all_results

# Usage example
async def priority_processing_example():
    agent = WebBrowsingAgent(api_key="your-key", headless=True)
    processor = PriorityBatchProcessor(agent)
    
    prioritized_tasks = [
        {"url": "https://critical-site.com", "task": "Extract", "priority": 10},
        {"url": "https://important-site.com", "task": "Extract", "priority": 5},
        {"url": "https://normal-site.com", "task": "Extract", "priority": 1},
        {"url": "https://low-priority-site.com", "task": "Extract", "priority": 0}
    ]
    
    results = await processor.run_priority_batch(prioritized_tasks)
    return results
```

---

## 🎯 Summary & Key Takeaways

The **Web Toolkit** represents the **scaling layer** of your Playwright LangGraph Agent system. Here are the essential concepts you should master:

### **🏗️ Architectural Principles**
✅ **Batch Orchestration**: Coordinate many individual agent tasks efficiently  
✅ **Resource Management**: Control concurrency, memory, and system resources  
✅ **Data Pipeline**: Transform raw agent results into actionable business intelligence  
✅ **Production Readiness**: Checkpointing, monitoring, and error recovery  

### **🛠️ Core Capabilities**
- **Concurrent Processing**: Execute multiple web automation tasks in parallel
- **Data Export**: Convert results to CSV, JSON, Excel, and custom formats
- **Error Recovery**: Retry logic, checkpointing, and graceful failure handling
- **Performance Optimization**: Concurrency tuning and resource monitoring
- **Enterprise Features**: Logging, monitoring, and production deployment

### **🚀 Production Scaling Patterns**
- **Domain Grouping**: Respectful crawling with per-domain rate limits
- **Memory Management**: Handle large datasets without memory exhaustion
- **Checkpointing**: Resume interrupted batches for long-running jobs
- **Multi-Agent Pools**: Scale throughput with multiple browser instances
- **Performance Monitoring**: Track and optimize batch processing metrics

### **📈 When to Use Web Toolkit**
- **Large-Scale Data Collection**: Process hundreds or thousands of URLs
- **Periodic Monitoring**: Scheduled price tracking, content monitoring
- **Research Automation**: Multi-source competitive intelligence gathering
- **Business Intelligence**: Automated market research and analysis
- **Production Workflows**: Enterprise-grade web automation pipelines

### **🎓 Mastery Progression**
1. **Start with Basic Batches**: Learn `run_batch()` and simple exports
2. **Add Error Handling**: Implement retry logic and graceful failures
3. **Optimize Performance**: Tune concurrency and monitor resource usage
4. **Scale to Production**: Add checkpointing, logging, and monitoring
5. **Build Custom Solutions**: Create domain-specific batch processors

### **💡 Best Practices**
- **Be Respectful**: Implement rate limiting and respect robots.txt
- **Monitor Resources**: Track memory usage and system performance
- **Plan for Failures**: Always include retry logic and error recovery
- **Export Everything**: Save results in multiple formats for different audiences
- **Document Thoroughly**: Log all operations for debugging and auditing

The Web Toolkit transforms your individual agent capabilities into **enterprise-scale automation systems**. It's the difference between processing a few web pages manually and automating business-critical data collection workflows that run reliably at scale.

**Start simple, scale smart, automate everything!** 🧰🚀

---

*This guide provides the foundation for mastering batch web automation. Use these patterns and examples as building blocks for your own large-scale web automation projects.*
    