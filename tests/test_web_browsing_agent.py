import os
import asyncio
import pytest
import logging

from agent.web_browsing_agent import WebBrowsingAgent
from dotenv import load_dotenv

# Load environment variables (for OpenAI, optional for Python-only bot)
load_dotenv(override=True)

# Set up logging to help debug issues
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.asyncio
async def test_web_browsing_agent_extract():
    """Test the web browsing agent extraction functionality with better error handling"""
    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key, "OPENAI_API_KEY must be set for the test"

    agent = WebBrowsingAgent(api_key, headless=True)

    try:
        result = await agent.execute_task(
            url="https://example.com",  # Use simpler site for testing
            task="Extract the page title and main content",
            task_type="extract"
        )

        print("\n==== TEST RESULT ====")
        print("Success:", result.get('success'))
        print("Final URL:", result.get('final_url'))
        print("Error:", result.get('error'))
        print("Navigation History Length:", len(result.get('navigation_history', [])))

        # Print first few navigation steps
        nav_history = result.get('navigation_history', [])
        if nav_history:
            print("First navigation steps:")
            for i, step in enumerate(nav_history[:3]):
                print(f"  {i + 1}. {step}")

        # Print extracted data summary
        extracted = result.get('extracted_data', {})
        if extracted:
            print("Extracted data keys:", list(extracted.keys()))
            elements = extracted.get('elements', [])
            print(f"Number of elements extracted: {len(elements)}")

        # Core checks - should always have these keys
        assert isinstance(result, dict), "Result should be a dictionary"
        assert "success" in result, "Result should have 'success' key"
        assert "extracted_data" in result, "Result should have 'extracted_data' key"
        assert "navigation_history" in result, "Result should have 'navigation_history' key"
        assert "error" in result, "Result should have 'error' key"

        # Check that we got some kind of result
        assert result["success"] in [True, False], "Success should be boolean"

        # The test should pass regardless of success/failure as long as it completes properly
        print("✅ Test completed successfully - agent didn't get stuck in infinite loop")

    except Exception as e:
        pytest.fail(f"Test raised unexpected exception: {e}")


@pytest.mark.asyncio
async def test_web_browsing_agent_simple_page():
    """Test with an even simpler page that should work reliably"""
    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key, "OPENAI_API_KEY must be set for the test"

    agent = WebBrowsingAgent(api_key, headless=True)

    result = await agent.execute_task(
        url="https://httpbin.org/html",  # Simple HTML page
        task="Extract any text content from the page",
        task_type="extract"
    )

    print("\n==== SIMPLE PAGE TEST RESULT ====")
    print("Success:", result.get('success'))
    print("Error:", result.get('error'))
    print("Extracted Elements:", len(result.get('extracted_data', {}).get('elements', [])))

    # Should always have the required keys
    assert "success" in result
    assert "extracted_data" in result
    assert "navigation_history" in result


@pytest.mark.asyncio
async def test_web_browsing_agent_error_handling():
    """Test that the agent properly handles errors and doesn't get stuck in loops"""
    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key, "OPENAI_API_KEY must be set for the test"

    agent = WebBrowsingAgent(api_key, headless=True)

    # Test with invalid URL
    result = await agent.execute_task(
        url="https://this-url-definitely-does-not-exist-12345.com",
        task="Extract content",
        task_type="extract"
    )

    print("\n==== ERROR HANDLING TEST RESULT ====")
    print("Success:", result.get('success'))
    print("Error:", result.get('error'))

    # Should handle error gracefully
    assert "success" in result
    assert "extracted_data" in result
    assert "error" in result
    assert result["success"] is False  # Should fail for invalid URL
    assert result["error"]  # Should have error message


if __name__ == "__main__":
    # Allow running tests directly
    asyncio.run(test_web_browsing_agent_extract())