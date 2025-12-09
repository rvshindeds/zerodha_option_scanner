import os
from dotenv import load_dotenv
from pathlib import Path

print("Current working directory:", Path.cwd())
print("Files here:", [p.name for p in Path.cwd().iterdir()])

env_path = Path.cwd() / ".env"
print(".env exists here?", env_path.exists())

load_dotenv(dotenv_path=env_path)

print("Key loaded:", bool(os.getenv("KITE_API_KEY")))
print("Secret loaded:", bool(os.getenv("KITE_API_SECRET")))
