#!/usr/bin/env python3
"""
launch_gradio.py

Quick launcher for the Gradio UI with environment setup and validation.
"""

import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'gradio',
        'playwright',
        'langgraph',
        'langchain',
        'langchain_openai',
        'pandas'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    return missing

def install_playwright_browsers():
    """Install Playwright browser drivers if needed"""
    try:
        print("🔄 Installing Playwright browsers...")
        subprocess.run([sys.executable, "-m", "playwright", "install"], check=True)
        print("✅ Playwright browsers installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Playwright browsers: {e}")
        return False

def setup_environment():
    """Setup environment and validate configuration"""
    print("🚀 Setting up Playwright LangGraph Agent Gradio UI...")
    
    # Check if we're in the right directory
    if not Path("agent/web_browsing_agent.py").exists():
        print("❌ Error: Please run this script from the project root directory")
        print("   Make sure you can see the 'agent/' folder in the current directory")
        return False
    
    # Check for .env file
    if not Path(".env").exists():
        print("⚠️  Warning: No .env file found")
        print("   You'll need to enter your OpenAI API key in the web interface")
        
        # Optionally create a template .env file
        create_env = input("   Create a template .env file? (y/n): ").lower().strip()
        if create_env == 'y':
            with open(".env", "w") as f:
                f.write("# OpenAI API Configuration\n")
                f.write("OPENAI_API_KEY=your-api-key-here\n")
                f.write("\n# Optional: Default model\n")
                f.write("DEFAULT_MODEL=gpt-4\n")
            print("✅ Template .env file created")
    
    # Check required packages
    missing_packages = check_requirements()
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("   Please install them with:")
        print("   pip install -r requirements_gradio.txt")
        return False
    
    # Install Playwright browsers
    if not install_playwright_browsers():
        return False
    
    # Create necessary directories
    directories = [
        "results",
        "results/gradio_sessions",
        "results/batch_runs", 
        "results/screenshots",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Environment setup complete!")
    return True

def launch_gradio_ui():
    """Launch the Gradio interface"""
    try:
        # Import and run the Gradio interface
        from gradio_ui import main
        print("🌐 Launching Gradio interface...")
        print("   Interface will be available at: http://localhost:7860")
        print("   Press Ctrl+C to stop")
        main()
    except ImportError as e:
        print(f"❌ Failed to import gradio_ui: {e}")
        print("   Make sure gradio_ui.py is in the current directory")
        return False
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
        return True
    except Exception as e:
        print(f"❌ Error launching Gradio UI: {e}")
        return False

def main():
    """Main launcher function"""
    print("=" * 60)
    print("🕷️  PLAYWRIGHT LANGGRAPH AGENT - GRADIO UI LAUNCHER")
    print("=" * 60)
    
    # Setup environment
    if not setup_environment():
        print("\n❌ Environment setup failed. Please fix the issues above and try again.")
        return 1
    
    print("\n" + "=" * 60)
    
    # Launch the UI
    if launch_gradio_ui():
        print("✅ Gradio UI launched successfully!")
        return 0
    else:
        print("❌ Failed to launch Gradio UI")
        return 1

if __name__ == "__main__":
    sys.exit(main())
