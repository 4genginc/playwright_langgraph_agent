#!/usr/bin/env python3
"""
test_api_key.py

Quick test script to validate your OpenAI API key
"""

import os
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_openai_connection():
    """Test OpenAI API connection"""
    try:
        from langchain_openai import ChatOpenAI
        
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            print("❌ No OPENAI_API_KEY found in environment")
            print("   Please set it in your .env file or environment variables")
            return False
        
        if not api_key.startswith('sk-'):
            print("❌ Invalid API key format (should start with 'sk-')")
            return False
        
        print(f"🔑 Testing API key: {api_key[:10]}...")
        
        # Test the connection
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",  # Use cheaper model for testing
            api_key=api_key,
            temperature=0.1,
            max_tokens=50
        )
        
        # Simple test query
        from langchain.schema import HumanMessage
        messages = [HumanMessage(content="Say 'Hello' if you can see this message.")]
        
        print("🔄 Testing OpenAI connection...")
        response = await llm.ainvoke(messages)
        
        print("✅ OpenAI API connection successful!")
        print(f"📝 Response: {response.content}")
        return True
        
    except ImportError as e:
        print(f"❌ Missing required packages: {e}")
        print("   Run: uv pip install langchain-openai")
        return False
    except Exception as e:
        print(f"❌ API connection failed: {e}")
        if "401" in str(e):
            print("   This usually means your API key is invalid")
        elif "429" in str(e):
            print("   Rate limit exceeded - try again in a moment")
        elif "insufficient_quota" in str(e):
            print("   Your API key has no remaining credits")
        return False

def test_agent_initialization():
    """Test agent initialization without API call"""
    try:
        from agent.web_browsing_agent import WebBrowsingAgent
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ No API key available for agent test")
            return False
        
        print("🤖 Testing agent initialization...")
        agent = WebBrowsingAgent(api_key, headless=True)
        print("✅ Agent initialization successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Cannot import agent: {e}")
        print("   Make sure you're in the correct directory")
        return False
    except Exception as e:
        print(f"❌ Agent initialization failed: {e}")
        return False

def main():
    print("=" * 60)
    print("🧪 OpenAI API Key & Agent Test")
    print("=" * 60)
    
    # Test 1: Check environment
    print("\n1️⃣ Testing environment setup...")
    env_file = ".env"
    if os.path.exists(env_file):
        print(f"✅ Found {env_file}")
    else:
        print(f"⚠️  No {env_file} file found")
        print("   You can still use environment variables")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print(f"✅ API key found: {api_key[:10]}...")
    else:
        print("❌ No OPENAI_API_KEY found")
        print("\nTo fix this:")
        print("1. Create a .env file with: OPENAI_API_KEY=your-key-here")
        print("2. Or export OPENAI_API_KEY=your-key-here")
        return 1
    
    # Test 2: Test API connection
    print("\n2️⃣ Testing OpenAI API connection...")
    try:
        api_success = asyncio.run(test_openai_connection())
    except Exception as e:
        print(f"❌ API test failed: {e}")
        api_success = False
    
    # Test 3: Test agent initialization
    print("\n3️⃣ Testing agent initialization...")
    agent_success = test_agent_initialization()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Test Summary:")
    print(f"   🔑 API Key: {'✅ Valid' if api_success else '❌ Failed'}")
    print(f"   🤖 Agent: {'✅ Ready' if agent_success else '❌ Failed'}")
    
    if api_success and agent_success:
        print("\n🎉 All tests passed! Your setup is ready.")
        print("   You can now run: python launch_gradio.py")
        return 0
    else:
        print("\n❌ Some tests failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    exit(main())
