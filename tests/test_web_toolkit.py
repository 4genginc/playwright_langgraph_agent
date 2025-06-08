def test_export_csv_and_json(tmp_path):
    from toolkit import web_toolkit
    dummy = [
        {"url": "https://a.com", "success": True, "data": {"k": 1}},
        {"url": "https://b.com", "success": False, "error": "not found"}
    ]
    csv_path = tmp_path / "dummy.csv"
    json_path = tmp_path / "dummy.json"
    web_toolkit.export_csv(dummy, csv_path)
    web_toolkit.export_json(dummy, json_path)
    assert csv_path.exists() and json_path.exists()