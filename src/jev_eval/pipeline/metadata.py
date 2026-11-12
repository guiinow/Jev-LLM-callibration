"""Provenance of a run: code version, machine, package and model versions.

Every lookup is best-effort: a missing tool or server records an error string instead of
failing the run.
"""
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from typing import Any

from jev_eval.config import EvalConfig
from jev_eval.env import REPO_ROOT

PACKAGES = ("typesafe-sdk", "openai", "numpy", "httpx")


def _cmd(*args: str) -> str | None:
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=10, cwd=REPO_ROOT)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _git() -> dict[str, Any]:
    status = _cmd("git", "status", "--porcelain")
    return {"commit": _cmd("git", "rev-parse", "HEAD"), "dirty": bool(status) if status is not None else None}


def _machine() -> dict[str, Any]:
    info: dict[str, Any] = {"platform": platform.platform(), "machine": platform.machine(),
                            "python": sys.version.split()[0]}
    if sys.platform == "darwin":
        memsize = _cmd("sysctl", "-n", "hw.memsize")
        info.update({"cpu": _cmd("sysctl", "-n", "machdep.cpu.brand_string"),
                     "hw_model": _cmd("sysctl", "-n", "hw.model"),
                     "memory_gb": round(int(memsize) / 2**30, 1) if memsize and memsize.isdigit() else None})
    return info


def _packages() -> dict[str, str | None]:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def ollama_root(base_url: str | None) -> str | None:
    """http://host:11434/v1 -> http://host:11434 (only for Ollama-looking endpoints)."""
    if not base_url or "11434" not in base_url:
        return None
    return base_url.rstrip("/").removesuffix("/v1")


def ollama_info(base_url: str | None, model: str | None) -> dict[str, Any] | None:
    root = ollama_root(base_url)
    if root is None:
        return None
    import httpx
    info: dict[str, Any] = {"url": root, "model": model}
    try:
        with httpx.Client(timeout=10) as client:
            info["version"] = client.get(f"{root}/api/version").json().get("version")
            tags = client.get(f"{root}/api/tags").json().get("models", [])
            tag = next((t for t in tags if t.get("name") == model or t.get("model") == model), None)
            info["digest"] = tag.get("digest") if tag else None
            show = client.post(f"{root}/api/show", json={"model": model}).json()
            info["details"] = show.get("details")
            info["parameters"] = show.get("parameters")  # e.g. "num_ctx 8192" from our Modelfile
            info["modelfile_from"] = next((line for line in (show.get("modelfile") or "").splitlines()
                                           if line.startswith("FROM")), None)
            info["context_length"] = next((v for k, v in (show.get("model_info") or {}).items()
                                           if k.endswith(".context_length")), None)
    except Exception as error:  # server down, old API: record it, don't block the run
        info["error"] = f"{type(error).__name__}: {error}"
    return info


def _redacted(settings: dict[str, Any]) -> dict[str, Any]:
    return {k: ("<redacted>" if "key" in k.lower() else v) for k, v in settings.items()}


def collect_run_metadata(config: EvalConfig, mock: bool) -> dict[str, Any]:
    llm = config.llm or {}
    return {
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mock": mock,
        "argv": sys.argv,
        "git": _git(),
        "config_path": str(config.config_path),
        "config": {"seed": config.seed, "sample_size": config.sample_size, "ece_bins": config.ece_bins,
                   "languages": config.languages, "systems": config.systems,
                   "jev": _redacted(config.jev or {}), "llm": _redacted(llm),
                   "paths": {k: str(v) for k, v in config.paths.items()}},
        "machine": _machine(),
        "packages": _packages(),
        "ollama": None if mock or "llm" not in config.systems else ollama_info(llm.get("base_url"), llm.get("model")),
    }


def append_run_metadata(path, entry: dict[str, Any]) -> None:
    """Keep one entry per `run` invocation, so resumed runs keep their full history."""
    history = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    history.append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
