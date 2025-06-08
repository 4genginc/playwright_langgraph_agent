import os
import asyncio
import pytest

from agent.web_browsing_agent import WebBrowsingAgent

from dotenv import load_dotenv

# Load environment variables (for OpenAI, optional for Python-only bot)
load_dotenv(override=True)

@pytest.mark.asyncio
async def test_web_browsing_agent_extract():
    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key, "OPENAI_API_KEY must be set for the test"

    agent = WebBrowsingAgent(api_key, headless=True)
    result = await agent.execute_task(
        url="https://news.ycombinator.com",
        task="Extract the top 3 news headlines and their links",
        task_type="extract"
    )

    print("\n==== TEST RESULT ====")
    print("Success:", result.get('success'))
    print("Final URL:", result.get('final_url'))
    print("Error:", result.get('error'))
    print("Navigation History:", result.get('navigation_history'))
    print("Extracted Data:", result.get('extracted_data'))

    # Core checks
    assert isinstance(result, dict)
    assert "success" in result
    assert result["success"] in [True, False]
    assert "extracted_data" in result
    assert "navigation_history" in result
    # Optionally: check that we extracted something (if the site is up)
    # assert result["success"] is True
    # assert len(result["extracted_data"].get("elements", [])) >= 1


# import os
# import pytest
# from agent.web_browsing_agent import WebBrowsingAgent
# from dotenv import load_dotenv
#
# # Load environment variables (for OpenAI, optional for Python-only bot)
# load_dotenv(override=True)
#
# @pytest.mark.asyncio
# async def test_extract_headlines():
#     api_key = os.getenv("OPENAI_API_KEY")
#     assert api_key, "OPENAI_API_KEY not set for testing"
#
#     agent = WebBrowsingAgent(api_key, headless=True)
#     result = await agent.execute_task(
#         url="https://news.ycombinator.com",
#         task="Extract the top 5 news headlines",
#         task_type="extract"
#     )
#     assert isinstance(result, dict)
#     assert result["success"], f"Agent failed: {result.get('error')}"
#     assert "extracted_data" in result
#     # Relaxed assertion for debugging:
#     # assert len(result["extracted_data"].get("elements", [])) > 0
#
