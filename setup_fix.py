#!/usr/bin/env python3
"""
setup_dependencies.py

Quick dependency setup script to resolve version conflicts
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} - Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed")
        print(f"   Error: {e.stderr}")
        return False

def setup_directories():
    """Create necessary directories"""
    directories = [
        "results",
        "results/screenshots", 
        "results/gradio_sessions",
        "results/batch_runs",
        "results/exports",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def main():
    print("🔧 Setting up Playwright LangGraph Agent Dependencies")
    print("=" * 60)
    
    # Check if we're in a virtual environment
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Warning: Not in a virtual environment")
        response = input("Continue anyway? (y/n): ").lower().strip()
        if response != 'y':
            print("Exiting. Please activate your virtual environment and try again.")
            return 1
    
    # Create directories
    print("\n📁 Creating project directories...")
    setup_directories()
    
    # Check if uv is available, otherwise use pip
    uv_available = run_command("uv --version", "Checking uv availability")
    pip_cmd = "uv pip" if uv_available else "pip"
    
    print(f"\n📦 Using package manager: {pip_cmd}")
    
    # Install core dependencies with compatible versions
    print("\n🔄 Installing core dependencies...")
    
    # Step 1: Install LangChain ecosystem with compatible versions
    langchain_deps = [
        "langchain-core>=0.3.43",
        "langchain>=0.3.0", 
        "langchain-openai>=0.2.0",
        "langgraph>=0.4.8"
    ]
    
    for dep in langchain_deps:
        if not run_command(f"{pip_cmd} install '{dep}'", f"Installing {dep}"):
            print(f"❌ Failed to install {dep}")
            return 1
    
    # Step 2: Install other core dependencies
    other_deps = [
        "playwright>=1.43.0",
        "pydantic>=2.6", 
        "python-dotenv>=1.0",
        "gradio>=4.0.0",
        "pandas>=2.0.0",
        "pillow>=10.0.0",
        "tqdm>=4.66",
        "nest_asyncio>=1.6"
    ]
    
    for dep in other_deps:
        if not run_command(f"{pip_cmd} install '{dep}'", f"Installing {dep}"):
            print(f"❌ Failed to install {dep}")
            return 1
    
    # Step 3: Install optional dependencies
    print("\n🎨 Installing optional dependencies...")
    optional_deps = [
        "matplotlib>=3.7.0",
        "plotly>=5.15.0", 
        "openpyxl>=3.1.0",
        "rich>=13.0.0",
        "pytest>=7.0.0",
        "pytest-asyncio>=0.21.0"
    ]
    
    for dep in optional_deps:
        run_command(f"{pip_cmd} install '{dep}'", f"Installing {dep} (optional)")
    
    # Install Playwright browsers
    print("\n🌐 Installing Playwright browsers...")
    if not run_command("playwright install", "Installing Playwright browsers"):
        print("❌ Failed to install Playwright browsers")
        return 1
    
    # Create .env file if it doesn't exist
    if not Path(".env").exists():
        print("\n📝 Creating .env template...")
        with open(".env", "w") as f:
            f.write("# OpenAI API Configuration\n")
            f.write("OPENAI_API_KEY=your-api-key-here\n")
            f.write("\n# Optional: Default model\n")
            f.write("DEFAULT_MODEL=gpt-4\n")
        print("✅ Created .env template - please add your OpenAI API key")
    
    print("\n" + "=" * 60)
    print("🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Add your OpenAI API key to the .env file")
    print("2. Run: python launch_gradio.py")
    print("3. Open http://localhost:7860 in your browser")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
