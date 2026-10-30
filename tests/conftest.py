import pytest


@pytest.fixture(autouse=True)
def _ignore_extra_keys_in_the_real_dotenv(monkeypatch):
    """`api_keys()` loads the project's git-ignored `.env`, so a developer who added GROQ_API_KEY_2/_3 would change what
    tests that expect a single key see. A blank value is ignored by `api_keys` and is never overridden by `load_env`;
    tests that want extra keys set them explicitly."""
    for n in range(2, 10):
        monkeypatch.setenv(f"GROQ_API_KEY_{n}", "")
