def test_env_loading(monkeypatch):
    import config
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key")
    assert config.get_api_key() == "dummy-key"
