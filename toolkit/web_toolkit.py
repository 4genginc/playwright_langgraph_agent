# import asyncio
# import json
# import csv
# import logging
# from agent.web_browsing_agent import WebBrowsingAgent

# logger = logging.getLogger(__name__)

# class WebBrowsingToolkit:
#     @staticmethod
#     async def batch_process_urls(api_key: str, url_tasks: list, max_concurrent: int = 3):
#         # ...

#     @staticmethod
#     def save_results_to_csv(results: list, filename: str = "web_browsing_results.csv"):
#         # ...


# toolkit/web_toolkit.py
import csv
import json

def export_csv(results, filename):
    keys = set()
    for res in results:
        keys.update(res.keys())
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(keys))
        writer.writeheader()
        writer.writerows(results)

