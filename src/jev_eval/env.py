import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_env(path: Path | None = None) -> None:
    """Load KEY=VALUE lines from the repo .env into os.environ (existing vars win)."""
    env_path = path or REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if value and key not in os.environ:
            os.environ[key] = value
