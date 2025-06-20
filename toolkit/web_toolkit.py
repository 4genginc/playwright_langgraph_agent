"""
toolkit/web_toolkit.py

Batch utilities for the Playwright LangGraph Agent:
- run_batch: Batch run agent tasks with concurrency control
- export_csv: Save results to CSV
- export_json: Save results to JSON
"""

import asyncio
import csv
import json

def export_csv(results, filename="results.csv"):
    """
    Save a list of dict results (from agent) to CSV.
    Handles nested dict/list fields by converting to JSON string.
    """
    if not results:
        return
    keys = {k for r in results for k in r}
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=sorted(keys))
        writer.writeheader()
        for r in results:
            row = {k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in r.items()}
            writer.writerow(row)

def export_json(results, filename="results.json"):
    """
    Save a list of dict results to JSON.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

async def run_batch(agent, url_tasks, max_concurrent=3):
    """
    Run agent on a list of {url, task, task_type, form_data} dicts in parallel (with a concurrency limit).
    Example url_tasks:
        [
            {"url": "https://example.com", "task": "Extract links", "task_type": "extract"},
            ...
        ]
    Returns list of results (same order as input).
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_one(task):
        async with semaphore:
            return await agent.execute_task(
                url=task['url'],
                task=task['task'],
                task_type=task.get('task_type', 'extract'),
                form_data=task.get('form_data')
            )

    results = await asyncio.gather(*(run_one(t) for t in url_tasks))
    return results