from __future__ import annotations

import argparse
import logging
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from rapc_net.utils.io import ensure_dir, write_text


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT.parent / "data"


class StepNotComplete(RuntimeError):
    """Raised when a script intentionally stops before producing its required artifact."""


@dataclass(frozen=True)
class ScriptContext:
    root: Path
    data_root: Path
    name: str
    logger: logging.Logger
    log_path: Path
    status_path: Path
    started_at: float


def setup_script(script_file: str) -> ScriptContext:
    name = Path(script_file).stem
    logs = ensure_dir(ROOT / "logs")
    status = ensure_dir(ROOT / "status")
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(logs / f"{name}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return ScriptContext(ROOT, DATA_ROOT, name, logger, logs / f"{name}.log", status / f"{name}.done", time.time())


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dry-run", action="store_true", help="Print planned work without writing heavy outputs.")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum samples to process for quick verification.")
    parser.add_argument("--dataset", default=None, help="Dataset name filter.")
    parser.add_argument("--model", default=None, choices=["grnet", "snowflakenet", "pointr", "attriflow", "rapc_net", "partial", "fusion"], help="Model filter.")
    parser.add_argument("--device", default="cuda", help="Torch device, for example cuda or cpu.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint where supported.")
    parser.add_argument("--quick", action="store_true", help="Run a quick sanity artifact step without enforcing later full-experiment prerequisites.")


def require_done(ctx: ScriptContext, previous: str | None, quick: bool = False) -> bool:
    if previous is None:
        return True
    marker = ctx.root / "status" / f"{previous}.done"
    if marker.exists():
        return True
    if quick:
        ctx.logger.warning("Quick mode: skipping missing prerequisite marker %s", marker)
        return True
    ctx.logger.error("Required previous step is missing: %s", marker)
    ctx.logger.error("Run first: python scripts\\%s.py", previous)
    return False


def mark_done(ctx: ScriptContext, summary: str) -> None:
    elapsed = time.time() - ctx.started_at
    write_text(ctx.status_path, f"{summary}\nelapsed_seconds={elapsed:.3f}\n")
    ctx.logger.info("Wrote status marker: %s", ctx.status_path)


def print_paths(ctx: ScriptContext) -> None:
    ctx.logger.info("Project root: %s", ctx.root)
    ctx.logger.info("Data root: %s", ctx.data_root)
    ctx.logger.info("Log path: %s", ctx.log_path)
    ctx.logger.info("Status path: %s", ctx.status_path)


def finish(ctx: ScriptContext, summary: str, next_script: str | None) -> None:
    mark_done(ctx, summary)
    ctx.logger.info("SUCCESS: %s", summary)
    if next_script:
        ctx.logger.info("Next command: python scripts\\%s.py", next_script)
    else:
        ctx.logger.info("This is the final numbered step.")


def fail(ctx: ScriptContext, message: str, remedy: str, code: int = 2) -> int:
    ctx.logger.error("FAILED: %s", message)
    ctx.logger.error("Suggested fix: %s", remedy)
    return code


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ModuleNotFoundError:
        return


def disk_free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024**3)


def git_commit(root: Path = ROOT) -> str:
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "not-a-git-repository"


def guarded_main(
    script_file: str,
    previous: str | None,
    next_script: str | None,
    build_parser: Callable[[argparse.ArgumentParser], None],
    run: Callable[[ScriptContext, argparse.Namespace], str],
) -> int:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_args(parser)
    build_parser(parser)
    args = parser.parse_args()
    ctx = setup_script(script_file)
    print_paths(ctx)
    seed_everything(args.seed)
    if not require_done(ctx, previous, quick=args.quick):
        return 2
    try:
        summary = run(ctx, args)
    except StepNotComplete as exc:
        ctx.logger.warning("NOT COMPLETE: %s", exc)
        ctx.logger.warning("No .done marker was written for %s.", ctx.name)
        ctx.logger.warning("Rerun this step with the required confirmation or inputs.")
        return 0
    except KeyboardInterrupt:
        ctx.logger.error("Interrupted by user. Saving safe stop marker.")
        write_text(ctx.root / "status" / f"{ctx.name}.interrupted", f"interrupted_at={time.time()}\n")
        return 130
    except Exception as exc:
        ctx.logger.exception("Unhandled error")
        ctx.logger.error("Remedy: inspect %s, fix the reported input/config issue, and rerun this exact script.", ctx.log_path)
        return 1
    finish(ctx, summary, next_script)
    return 0


def bool_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}
