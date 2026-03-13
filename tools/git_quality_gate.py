#!/usr/bin/env python
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], env: dict[str, str]) -> int:
    print(f"[quality-gate] running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
    return proc.returncode


def run_pyright(env: dict[str, str]) -> int:
    return run_pyright_for_files(env, files=None)


def run_pyright_for_files(env: dict[str, str], files: list[str] | None) -> int:
    pyright_bin = shutil.which("pyright")
    if pyright_bin:
        cmd = [pyright_bin]
        if files:
            cmd.extend(files)
        return run(cmd, env)

    if importlib.util.find_spec("pyright") is None:
        print(
            "[quality-gate] pyright not found. Install it in your environment, "
            "e.g. 'pip install pyright'.",
            file=sys.stderr,
        )
        return 1

    cmd = [sys.executable, "-m", "pyright"]
    if files:
        cmd.extend(files)
    return run(cmd, env)


def run_pytest(env: dict[str, str]) -> int:
    marker_expr = env.get("TRANSLATOR2_PYTEST_MARK_EXPR", "not slow").strip()
    cmd = [sys.executable, "-m", "pytest", "-q"]
    if marker_expr:
        cmd.extend(["-m", marker_expr])
    return run(cmd, env)


def staged_python_files(env: dict[str, str]) -> list[str]:
    proc = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return []

    files: list[str] = []
    for line in proc.stdout.splitlines():
        rel_path = line.strip()
        if not rel_path or not rel_path.endswith(".py"):
            continue
        if (REPO_ROOT / rel_path).exists():
            files.append(rel_path)
    return files


def changed_python_files_for_push(env: dict[str, str]) -> list[str]:
    upstream_proc = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    if upstream_proc.returncode == 0:
        diff_cmd = [
            "git",
            "diff",
            "--name-only",
            "--diff-filter=ACMR",
            f"{upstream_proc.stdout.strip()}..HEAD",
        ]
    else:
        diff_cmd = ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"]

    proc = subprocess.run(
        diff_cmd,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return []

    files: list[str] = []
    for line in proc.stdout.splitlines():
        rel_path = line.strip()
        if not rel_path or not rel_path.endswith(".py"):
            continue
        if not rel_path.startswith(("src/", "tests/", "tools/")):
            continue
        if (REPO_ROOT / rel_path).exists():
            files.append(rel_path)
    return files


def main() -> int:
    stage = sys.argv[1] if len(sys.argv) > 1 else "pre-push"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")

    if stage == "pre-commit":
        files = staged_python_files(env)
        if not files:
            print(
                "[quality-gate] pre-commit: no staged Python files, skipping pyright."
            )
            return 0
        if run_pyright_for_files(env, files) != 0:
            return 1
        return 0

    if stage == "pre-push":
        files = changed_python_files_for_push(env)
        if not files:
            print(
                "[quality-gate] pre-push: no changed Python files, "
                "skipping pyright and pytest."
            )
            return 0
        if run_pyright(env) != 0:
            return 1
        if run_pytest(env) != 0:
            return 1
        return 0

    print(f"[quality-gate] unknown stage: {stage}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
