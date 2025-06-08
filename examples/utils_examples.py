#!/usr/bin/env python3
"""
Practical examples of using utils.py in the Playwright LangGraph Agent project
Run these examples to see the utilities in action!
"""

from utils import (
    ensure_dir, timestamp_str, save_json, load_json, 
    setup_basic_logging, print_banner, die
)
import os
import logging

# Example 1: Project Setup Script
def setup_project():
    """Initialize project directory structure and configuration"""
    print_banner("🚀 Project Setup", char="=", width=50)
    
    # Create all necessary directories
    directories = [
        "results",
        "results/screenshots", 
        "results/batch_runs",
        "results/exports",
        "logs",
        "config",
        "temp"
    ]
    
    for directory in directories:
        ensure_dir(directory)
        print(f"✅ Created directory: {directory}")
    
    # Create default configuration file
    default_config = {
        "agent_settings": {
            "headless": True,
            "viewport": {"width": 1280, "height": 720},
            "timeout": 30000,
            "max_retries": 3
        },
        "export_settings": {
            "format": "json",
            "include_metadata": True,
            "timestamp_files": True
        },
        "logging": {
            "level": "INFO",
            "file": "logs/agent.log"
        }
    }
    
    config_file = "config/settings.json"
    save_json(default_config, config_file)
    print(f"✅ Created config file: {config_file}")
    
    print_banner("✨ Setup Complete!")

# Example 2: Batch Processing with Proper Logging
def run_batch_demo():
    """Demonstrate batch processing with logging and result management"""
    
    # Setup logging first
    timestamp = timestamp_str()
    log_file = f"logs/batch_{timestamp}.log"
    setup_basic_logging("INFO", log_file)
    logger = logging.getLogger(__name__)
    
    print_banner("📊 Batch Processing Demo")
    
    # Prepare batch results directory
    batch_dir = f"results/batch_runs/batch_{timestamp}"
    ensure_dir(batch_dir)
    
    # Simulate processing multiple URLs
    urls = [
        "https://example.com",
        "https://httpbin.org",
        "https://jsonplaceholder.typicode.com"
    ]
    
    batch_results = []
    
    for i, url in enumerate(urls, 1):
        logger.info(f"Processing {i}/{len(urls)}: {url}")
        print(f"🔄 Processing {i}/{len(urls)}: {url}")
        
        # Simulate agent work (replace with actual agent call)
        result = {
            "url": url,
            "processed_at": timestamp_str(),
            "batch_id": timestamp,
            "sequence": i,
            "success": True,
            "extracted_data": {
                "title": f"Sample Title {i}",
                "links_found": i * 3,
                "processing_time": 1.5 + i * 0.3
            }
        }
        
        # Save individual result
        result_file = f"{batch_dir}/result_{i:03d}.json"
        save_json(result, result_file)
        batch_results.append(result)
        
        logger.info(f"Saved result to {result_file}")
    
    # Save consolidated batch results
    batch_summary = {
        "batch_id": timestamp,
        "total_urls": len(urls),
        "successful": len(batch_results),
        "failed": 0,
        "results": batch_results,
        "batch_completed_at": timestamp_str()
    }
    
    summary_file = f"{batch_dir}/batch_summary.json"
    save_json(batch_summary, summary_file)
    
    print_banner(f"✅ Batch Complete! Results in {batch_dir}")
    logger.info(f"Batch processing complete. Summary saved to {summary_file}")

# Example 3: Data Analysis and Reporting
def analyze_results():
    """Analyze previous batch results and generate reports"""
    print_banner("📈 Results Analysis")
    
    # Try to load existing batch results
    import glob
    
    batch_dirs = glob.glob("results/batch_runs/batch_*")
    if not batch_dirs:
        print("❌ No batch results found. Run batch demo first!")
        return
    
    # Analyze the most recent batch
    latest_batch = max(batch_dirs)
    print(f"🔍 Analyzing: {latest_batch}")
    
    try:
        summary_file = f"{latest_batch}/batch_summary.json"
        batch_data = load_json(summary_file)
        
        # Generate analysis
        analysis = {
            "analysis_timestamp": timestamp_str(),
            "batch_analyzed": batch_data["batch_id"],
            "total_processed": batch_data["total_urls"],
            "success_rate": batch_data["successful"] / batch_data["total_urls"],
            "average_processing_time": sum(
                r["extracted_data"]["processing_time"] 
                for r in batch_data["results"]
            ) / len(batch_data["results"]),
            "total_links_found": sum(
                r["extracted_data"]["links_found"] 
                for r in batch_data["results"]
            )
        }
        
        # Save analysis
        analysis_file = f"{latest_batch}/analysis_{timestamp_str()}.json"
        save_json(analysis, analysis_file)
        
        # Print summary
        print(f"📊 Success Rate: {analysis['success_rate']:.1%}")
        print(f"⏱️  Avg Processing Time: {analysis['average_processing_time']:.2f}s")
        print(f"🔗 Total Links Found: {analysis['total_links_found']}")
        print(f"💾 Analysis saved to: {analysis_file}")
        
    except FileNotFoundError as e:
        die(f"Could not load batch summary: {e}")
    except Exception as e:
        die(f"Analysis failed: {e}")

# Example 4: Environment Validation
def validate_environment():
    """Validate that the environment is properly configured"""
    print_banner("🔧 Environment Validation")
    
    # Check required environment variables
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
        else:
            print(f"✅ {var}: Found")
    
    if missing_vars:
        die(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    # Check required directories exist
    required_dirs = ["results", "logs", "config"]
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✅ Directory {directory}: Exists")
        else:
            print(f"⚠️  Directory {directory}: Missing (will create)")
            ensure_dir(directory)
    
    # Check configuration file
    config_file = "config/settings.json"
    if os.path.exists(config_file):
        try:
            config = load_json(config_file)
            print(f"✅ Configuration: Loaded {len(config)} sections")
        except Exception as e:
            die(f"Configuration file is corrupted: {e}")
    else:
        print("⚠️  Configuration file not found")
    
    print_banner("✅ Environment Valid!")

# Example 5: Cleanup and Maintenance
def cleanup_old_results(days_old=7):
    """Clean up old result files"""
    import time
    import glob
    
    print_banner("🧹 Cleanup Old Results")
    
    cutoff_time = time.time() - (days_old * 24 * 60 * 60)
    
    # Find old batch directories
    batch_dirs = glob.glob("results/batch_runs/batch_*")
    old_dirs = []
    
    for batch_dir in batch_dirs:
        dir_time = os.path.getctime(batch_dir)
        if dir_time < cutoff_time:
            old_dirs.append(batch_dir)
    
    if not old_dirs:
        print(f"✅ No directories older than {days_old} days found")
        return
    
    # Create backup before cleanup
    backup_file = f"results/cleanup_backup_{timestamp_str()}.json"
    backup_data = {
        "cleanup_timestamp": timestamp_str(),
        "days_threshold": days_old,
        "directories_removed": old_dirs
    }
    save_json(backup_data, backup_file)
    
    # Remove old directories (simulated - uncomment to actually delete)
    for old_dir in old_dirs:
        print(f"🗑️  Would remove: {old_dir}")
        # shutil.rmtree(old_dir)  # Uncomment to actually delete
    
    print(f"🔄 Cleanup plan saved to: {backup_file}")
    print(f"📁 {len(old_dirs)} directories marked for cleanup")

# Main demonstration function
def main():
    """Run all utility demonstrations"""
    try:
        # Run all examples in sequence
        setup_project()
        validate_environment()
        run_batch_demo()
        analyze_results()
        cleanup_old_results()
        
        print_banner("🎉 All Examples Complete!", char="🎊", width=60)
        
    except Exception as e:
        die(f"Demo failed: {e}")

if __name__ == "__main__":
    main()
