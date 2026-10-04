import os

# Never let unit tests pick up a developer's .env or real credentials.
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("JWT_SECRET", "")
