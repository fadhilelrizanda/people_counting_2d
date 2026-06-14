"""
Kaggle MCP Server for Claude Code

Wraps the `kaggle` CLI (v2.2.1) as MCP tools so Claude can:
  - Browse and download competitions
  - Search and download datasets
  - Push/pull/list kernels (notebooks)
  - Manage models
  - View config and quota

Each tool shells out to `kaggle <group> <command> ...` and returns the
stdout text (or an error message).
"""

import json
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

mcp = FastMCP("kaggle", instructions="Tools that wrap the Kaggle CLI.")


def _run(
    command: list[str],
    timeout: int = 120,
    env: Optional[dict[str, str]] = None,
) -> str:
    """Run *command* via subprocess and return stripped stdout.

    Args:
        command: The command and arguments.
        timeout: Maximum runtime in seconds.
        env: Optional extra environment variables merged into the subprocess
            environment (overrides inherited values).
    """
    cmd_env = os.environ.copy()
    if env:
        cmd_env.update(env)
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=cmd_env,
        )
    except subprocess.TimeoutExpired:
        return f"[timeout after {timeout}s]"
    except FileNotFoundError:
        return (
            "[error] `kaggle` CLI not found on PATH.  "
            "Install it with: pip install kaggle"
        )

    if proc.returncode != 0:
        # The kaggle CLI writes errors to stdout in some cases, but
        # stderr is the more reliable channel.
        msg = (proc.stderr or proc.stdout or "").strip()
        return f"[exit code {proc.returncode}]\n{msg}"

    return proc.stdout.strip() or "(empty response)"


# ---------------------------------------------------------------------------
# Multi-user Profiles
# ---------------------------------------------------------------------------


@dataclass
class Profile:
    name: str
    token: str


_PROFILES_FILE = Path.home() / ".kaggle" / "profiles.json"
_profiles: dict[str, Profile] = {}
_active_profile: str | None = None
_lock = threading.Lock()


def _load_profiles() -> None:
    """Load profiles from disk, migrating the legacy token if needed."""
    global _active_profile

    if _PROFILES_FILE.exists():
        raw = json.loads(_PROFILES_FILE.read_text())
        profiles: dict[str, Profile] = {}
        for p in raw.get("profiles", []):
            profiles[p["name"]] = Profile(name=p["name"], token=p["token"])
        _profiles.update(profiles)
        _active_profile = raw.get("default", list(profiles.keys())[0] if profiles else None)
        return

    # First run — migrate from existing sources.
    profiles = {}
    token: str | None = None

    # Priority 1: KAGGLE_API_TOKEN env var (set in .env or shell).
    env_token = os.environ.get("KAGGLE_API_TOKEN")
    if env_token:
        # The env var might point to a file path — check, otherwise use literal.
        p = Path(env_token)
        token = p.read_text().strip() if p.exists() else env_token

    # Priority 2: ~/.kaggle/access_token file.
    if not token:
        token_file = Path.home() / ".kaggle" / "access_token"
        if token_file.exists():
            token = token_file.read_text().strip()

    if token:
        profiles["default"] = Profile(name="default", token=token)
        _save_profiles(profiles, "default")

    _profiles.update(profiles)
    _active_profile = "default" if "default" in profiles else None


def _save_profiles(
    profiles: dict[str, Profile] | None = None,
    default: str | None = None,
) -> None:
    """Persist profiles to disk."""
    ps = profiles or _profiles
    data = {
        "profiles": [
            {"name": p.name, "token": p.token}
            for p in sorted(ps.values(), key=lambda x: x.name)
        ],
    }
    if default is None and _active_profile:
        data["default"] = _active_profile
    elif default:
        data["default"] = default
    _PROFILES_FILE.parent.mkdir(parents=True, exist_ok=True)
    _PROFILES_FILE.write_text(json.dumps(data, indent=2))
    _PROFILES_FILE.chmod(0o600)


def _resolve_token(profile_name: str | None = None) -> str | None:
    """Get the API token for *profile_name* (or the active profile).

    Returns None when no profile or token is configured.
    """
    with _lock:
        name = profile_name or _active_profile
        if name and name in _profiles:
            return _profiles[name].token
        return None


def _run_as(command: list[str], timeout: int = 120) -> str:
    """Run *command* with the active profile's API token injected."""
    token = _resolve_token()
    env = {"KAGGLE_API_TOKEN": token} if token else None
    return _run(command, timeout=timeout, env=env)


# Initialize the profile store on module import.
_load_profiles()


# ---------------------------------------------------------------------------
# Profile Management
# ---------------------------------------------------------------------------


@mcp.tool()
def kaggle_profiles_list() -> str:
    """List all configured profiles with the active one marked."""
    with _lock:
        if not _profiles:
            return "No profiles configured."

        lines = ["Profile list:"]
        for p in sorted(_profiles.values(), key=lambda x: x.name):
            marker = " (active)" if p.name == _active_profile else ""
            token_masked = p.token[:12] + "..." + p.token[-4:] if len(p.token) > 20 else "***masked***"
            lines.append(f"  - {p.name}{marker}  [{token_masked}]")
        return "\n".join(lines)


@mcp.tool()
def kaggle_profile_add(name: str, token: str) -> str:
    """Add a new Kaggle API profile.

    Args:
        name: A short label for this profile (e.g. 'work', 'personal').
        token: The Kaggle API token (KGAT_...).
    """
    if not token.startswith("KGAT_"):
        return "[error] Invalid token format — expected a KGAT_... token."

    with _lock:
        if name in _profiles:
            return f"[error] Profile '{name}' already exists. Remove it first or use a different name."

        _profiles[name] = Profile(name=name, token=token)
        _save_profiles()
    return f"Profile '{name}' added."


@mcp.tool()
def kaggle_profile_remove(name: str) -> str:
    """Remove a Kaggle API profile.

    Args:
        name: The profile label to remove.
    """
    global _active_profile

    with _lock:
        if name not in _profiles:
            return f"[error] Profile '{name}' not found."

        del _profiles[name]

        # If we removed the active profile, switch to the next available one.
        if _active_profile == name:
            _active_profile = next(
                (p.name for p in sorted(_profiles.values(), key=lambda x: x.name)),
                None,
            )

        _save_profiles()
    return f"Profile '{name}' removed."


@mcp.tool()
def kaggle_profile_use(name: str) -> str:
    """Switch the active Kaggle profile.

    Subsequent tool calls will use the token of this profile.

    Args:
        name: The profile label to activate.
    """
    global _active_profile

    with _lock:
        if name not in _profiles:
            return f"[error] Profile '{name}' not found. Available: {', '.join(sorted(_profiles))}"

        _active_profile = name
        _save_profiles()
    return f"Switched to profile '{name}'."


@mcp.tool()
def kaggle_profile_show() -> str:
    """Show the currently active profile name."""
    with _lock:
        if _active_profile:
            return _active_profile
        return "No active profile (no profiles configured)."


# ---------------------------------------------------------------------------
# Competitions
# ---------------------------------------------------------------------------


@mcp.tool()
def kaggle_competitions_list(
    group: Optional[str] = None,
    category: Optional[str] = None,
    sort_by: Optional[str] = None,
    search: Optional[str] = None,
    page: Optional[int] = None,
    page_size: Optional[int] = None,
    page_token: Optional[str] = None,
    csv: bool = False,
) -> str:
    """List available Kaggle competitions.

    Args:
        group: Competition group — 'general' (default), 'entered', or 'inClass'.
        category: Category filter — 'all', 'featured', 'research', 'recruitment',
            'gettingStarted', 'masters', or 'playground'.
        sort_by: Sort order — 'latestDeadline' (default), 'grouped', 'prize',
            'earliestDeadline', 'numberOfTeams', or 'recentlyCreated'.
        search: Free-text search term.
        page: Page number (1-based, page size 20 by default).
        page_size: Items per page (max 200).
        page_token: Opaque page token for cursor-based paging.
        csv: Print results in CSV format instead of a table.
    """
    cmd = ["kaggle", "competitions", "list"]
    if group:
        cmd.extend(["--group", group])
    if category:
        cmd.extend(["--category", category])
    if sort_by:
        cmd.extend(["--sort-by", sort_by])
    if search:
        cmd.extend(["--search", search])
    if page is not None:
        cmd.extend(["--page", str(page)])
    if page_size is not None:
        cmd.extend(["--page-size", str(page_size)])
    if page_token:
        cmd.extend(["--page-token", page_token])
    if csv:
        cmd.append("-v")
    return _run_as(cmd)


@mcp.tool()
def kaggle_competitions_files(
    competition: Optional[str] = None,
    csv: bool = False,
    page_size: Optional[int] = None,
    page_token: Optional[str] = None,
) -> str:
    """List files available for download in a competition.

    Args:
        competition: Competition URL suffix (e.g. 'titanic'). Uses the default
            competition from config if omitted.
        csv: Print results in CSV format.
        page_size: Items per page (max 200).
        page_token: Opaque page token for cursor-based paging.
    """
    cmd = ["kaggle", "competitions", "files"]
    if competition:
        cmd.append(competition)
    if csv:
        cmd.append("-v")
    if page_size is not None:
        cmd.extend(["--page-size", str(page_size)])
    if page_token:
        cmd.extend(["--page-token", page_token])
    return _run_as(cmd)


@mcp.tool()
def kaggle_competitions_download(
    competition: str,
    file_name: Optional[str] = None,
    path: Optional[str] = None,
    force: bool = False,
    quiet: bool = False,
) -> str:
    """Download competition files.

    Args:
        competition: Competition URL suffix (e.g. 'titanic').
        file_name: Download a specific file by name. Defaults to all files.
        path: Folder where files will be downloaded (defaults to current
            working directory).
        force: Skip up-to-date check and force download.
        quiet: Suppress progress output.
    """
    cmd = ["kaggle", "competitions", "download", competition]
    if file_name:
        cmd.extend(["-f", file_name])
    if path:
        cmd.extend(["-p", path])
    if force:
        cmd.append("-o")
    if quiet:
        cmd.append("-q")
    return _run_as(cmd)


@mcp.tool()
def kaggle_competitions_submit(
    competition: str,
    message: str,
    file_name: Optional[str] = None,
    kernel: Optional[str] = None,
    version: Optional[str] = None,
    quiet: bool = False,
) -> str:
    """Submit to a Kaggle competition.

    Provide either `file_name` (for file-based submissions) OR `kernel` +
    optional `version` (for code competitions), not both.

    Args:
        competition: Competition URL suffix (e.g. 'titanic').
        message: Submission description (required).
        file_name: Full path to the submission file, or the name of a kernel
            output file for code competitions.
        kernel: Name of the kernel (notebook) to submit for code competitions.
        version: Kernel version number (e.g. '3').
        quiet: Suppress progress output.
    """
    cmd = ["kaggle", "competitions", "submit", "-m", message]
    if competition:
        cmd.append(competition)
    if file_name:
        cmd.extend(["-f", file_name])
    if kernel:
        cmd.extend(["-k", kernel])
    if version:
        cmd.extend(["-v", version])
    if quiet:
        cmd.append("-q")
    return _run_as(cmd)


@mcp.tool()
def kaggle_competitions_submissions(
    competition: Optional[str] = None,
    csv: bool = False,
    page_size: Optional[int] = None,
    page_token: Optional[str] = None,
) -> str:
    """Show your competition submissions.

    Args:
        competition: Competition URL suffix.
        csv: Print results in CSV format.
        page_size: Items per page (max 200).
        page_token: Opaque page token for cursor-based paging.
    """
    cmd = ["kaggle", "competitions", "submissions"]
    if competition:
        cmd.append(competition)
    if csv:
        cmd.append("-v")
    if page_size is not None:
        cmd.extend(["--page-size", str(page_size)])
    if page_token:
        cmd.extend(["--page-token", page_token])
    return _run_as(cmd)


@mcp.tool()
def kaggle_competitions_leaderboard(
    competition: Optional[str] = None,
    show: bool = False,
    download: bool = False,
    path: Optional[str] = None,
    csv: bool = False,
    page_size: Optional[int] = None,
    page_token: Optional[str] = None,
) -> str:
    """Get competition leaderboard information.

    Args:
        competition: Competition URL suffix.
        show: Show the top of the leaderboard inline.
        download: Download the full leaderboard as a CSV file.
        path: Folder for the downloaded leaderboard file.
        csv: Print results in CSV format.
        page_size: Items per page (max 200).
        page_token: Opaque page token for cursor-based paging.
    """
    cmd = ["kaggle", "competitions", "leaderboard"]
    if competition:
        cmd.append(competition)
    if show:
        cmd.append("-s")
    if download:
        cmd.append("-d")
    if path:
        cmd.extend(["-p", path])
    if csv:
        cmd.append("-v")
    if page_size is not None:
        cmd.extend(["--page-size", str(page_size)])
    if page_token:
        cmd.extend(["--page-token", page_token])
    return _run_as(cmd)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


@mcp.tool()
def kaggle_datasets_list(
    search: Optional[str] = None,
    sort_by: Optional[str] = None,
    file_type: Optional[str] = None,
    license_name: Optional[str] = None,
    tags: Optional[str] = None,
    mine: bool = False,
    user: Optional[str] = None,
    page: Optional[int] = None,
    max_size: Optional[int] = None,
    min_size: Optional[int] = None,
    csv: bool = False,
) -> str:
    """List available Kaggle datasets.

    Args:
        search: Free-text search terms.
        sort_by: Sort order — 'hottest' (default), 'votes', 'updated', or
            'active'.
        file_type: File type filter — 'all', 'csv', 'sqlite', 'json', or
            'bigQuery'.
        license_name: License filter — 'all', 'cc', 'gpl', 'odb', or 'other'.
        tags: Comma-separated tag IDs to filter by.
        mine: Show only my datasets.
        user: Show datasets owned by a specific user or organization.
        page: Page number (1-based, page size 20).
        max_size: Maximum dataset file size in bytes.
        min_size: Minimum dataset file size in bytes.
        csv: Print results in CSV format.
    """
    cmd = ["kaggle", "datasets", "list"]
    if search:
        cmd.extend(["--search", search])
    if sort_by:
        cmd.extend(["--sort-by", sort_by])
    if file_type:
        cmd.extend(["--file-type", file_type])
    if license_name:
        cmd.extend(["--license", license_name])
    if tags:
        cmd.extend(["--tags", tags])
    if mine:
        cmd.append("-m")
    if user:
        cmd.extend(["--user", user])
    if page is not None:
        cmd.extend(["-p", str(page)])
    if max_size is not None:
        cmd.extend(["--max-size", str(max_size)])
    if min_size is not None:
        cmd.extend(["--min-size", str(min_size)])
    if csv:
        cmd.append("-v")
    return _run_as(cmd)


@mcp.tool()
def kaggle_datasets_files(
    dataset: str,
    csv: bool = False,
    page_size: Optional[int] = None,
    page_token: Optional[str] = None,
) -> str:
    """List dataset files.

    Args:
        dataset: Dataset URL suffix in format <owner>/<dataset-name>.
        csv: Print results in CSV format.
        page_size: Items per page (max 200).
        page_token: Opaque page token for cursor-based paging.
    """
    cmd = ["kaggle", "datasets", "files", dataset]
    if csv:
        cmd.append("-v")
    if page_size is not None:
        cmd.extend(["--page-size", str(page_size)])
    if page_token:
        cmd.extend(["--page-token", page_token])
    return _run_as(cmd)


@mcp.tool()
def kaggle_datasets_download(
    dataset: str,
    file_name: Optional[str] = None,
    path: Optional[str] = None,
    unzip: bool = False,
    force: bool = False,
    quiet: bool = False,
) -> str:
    """Download dataset files.

    Args:
        dataset: Dataset URL suffix in format <owner>/<dataset-name>.
        file_name: Download a specific file. Defaults to all files.
        path: Folder where files will be downloaded.
        unzip: Unzip the downloaded file and remove the zip.
        force: Skip up-to-date check and force download.
        quiet: Suppress progress output.
    """
    cmd = ["kaggle", "datasets", "download", dataset]
    if file_name:
        cmd.extend(["-f", file_name])
    if path:
        cmd.extend(["-p", path])
    if unzip:
        cmd.append("--unzip")
    if force:
        cmd.append("-o")
    if quiet:
        cmd.append("-q")
    return _run_as(cmd)


@mcp.tool()
def kaggle_datasets_create(
    path: Optional[str] = None,
    public: bool = False,
    quiet: bool = False,
    keep_tabular: bool = False,
    dir_mode: Optional[str] = None,
) -> str:
    """Create a new Kaggle dataset.

    The folder must contain a valid dataset-metadata.json file.  See
    https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets_metadata.md

    Args:
        path: Folder with data files + dataset-metadata.json (defaults to
            current working directory).
        public: Create publicly (default is private).
        quiet: Suppress progress output.
        keep_tabular: Don't convert tabular files to CSV.
        dir_mode: What to do with directories — 'skip', 'zip', or 'tar'.
    """
    cmd = ["kaggle", "datasets", "create"]
    if path:
        cmd.extend(["-p", path])
    if public:
        cmd.append("-u")
    if quiet:
        cmd.append("-q")
    if keep_tabular:
        cmd.append("-t")
    if dir_mode:
        cmd.extend(["-r", dir_mode])
    return _run_as(cmd)


@mcp.tool()
def kaggle_datasets_version(
    message: str,
    path: Optional[str] = None,
    quiet: bool = False,
    keep_tabular: bool = False,
    dir_mode: Optional[str] = None,
    delete_old_versions: bool = False,
) -> str:
    """Create a new dataset version.

    Args:
        message: Version notes describing the changes (required).
        path: Folder with data files + dataset-metadata.json.
        quiet: Suppress progress output.
        keep_tabular: Don't convert tabular files to CSV.
        dir_mode: What to do with directories — 'skip', 'zip', or 'tar'.
        delete_old_versions: Delete old versions of this dataset.
    """
    cmd = ["kaggle", "datasets", "version", "-m", message]
    if path:
        cmd.extend(["-p", path])
    if quiet:
        cmd.append("-q")
    if keep_tabular:
        cmd.append("-t")
    if dir_mode:
        cmd.extend(["-r", dir_mode])
    if delete_old_versions:
        cmd.append("-d")
    return _run_as(cmd)


@mcp.tool()
def kaggle_datasets_status(
    dataset: str,
    fmt: Optional[str] = None,
) -> str:
    """Get the creation status of a dataset.

    Args:
        dataset: Dataset URL suffix in format <owner>/<dataset-name>.
        fmt: Output format. Defaults to plain text. Use 'json' to get a JSON
            object, or field-selection like 'json(status)'.
    """
    cmd = ["kaggle", "datasets", "status", dataset]
    if fmt:
        cmd.extend(["--format", fmt])
    return _run_as(cmd)


@mcp.tool()
def kaggle_datasets_metadata(
    dataset: str,
    path: Optional[str] = None,
    update: bool = False,
) -> str:
    """Download dataset metadata.

    Args:
        dataset: Dataset URL suffix in format <owner>/<dataset-name>.
        path: Location to download metadata to (defaults to current working
            directory).
        update: Update the dataset metadata file instead of downloading.
    """
    cmd = ["kaggle", "datasets", "metadata", dataset]
    if path:
        cmd.extend(["-p", path])
    if update:
        cmd.append("--update")
    return _run_as(cmd)


# ---------------------------------------------------------------------------
# Kernels (Notebooks / Scripts)
# ---------------------------------------------------------------------------


@mcp.tool()
def kaggle_kernels_list(
    mine: bool = False,
    page: Optional[int] = None,
    page_size: Optional[int] = None,
    search: Optional[str] = None,
    parent: Optional[str] = None,
    competition: Optional[str] = None,
    dataset: Optional[str] = None,
    user: Optional[str] = None,
    language: Optional[str] = None,
    kernel_type: Optional[str] = None,
    output_type: Optional[str] = None,
    sort_by: Optional[str] = None,
    csv: bool = False,
) -> str:
    """List Kaggle kernels (notebooks/scripts).

    Args:
        mine: Show only my kernels.
        page: Page number (1-based).
        page_size: Items per page (max 200).
        search: Free-text search term.
        parent: Find children of the specified parent kernel.
        competition: Find kernels for a given competition slug.
        dataset: Find kernels for a dataset (format <user>/<dataset-slug>).
        user: Find kernels by a specific user.
        language: Language — 'all', 'python', 'r', 'sqlite', or 'julia'.
        kernel_type: Type — 'all', 'script', or 'notebook'.
        output_type: Output type — 'all', 'visualizations', or 'data'.
        sort_by: Sort order — 'hotness' (default), 'commentCount',
            'dateCreated', 'dateRun', 'relevance', 'scoreAscending',
            'scoreDescending', 'viewCount', 'voteCount'.
        csv: Print results in CSV format.
    """
    cmd = ["kaggle", "kernels", "list"]
    if mine:
        cmd.append("-m")
    if page is not None:
        cmd.extend(["-p", str(page)])
    if page_size is not None:
        cmd.extend(["--page-size", str(page_size)])
    if search:
        cmd.extend(["-s", search])
    if parent:
        cmd.extend(["--parent", parent])
    if competition:
        cmd.extend(["--competition", competition])
    if dataset:
        cmd.extend(["--dataset", dataset])
    if user:
        cmd.extend(["--user", user])
    if language:
        cmd.extend(["--language", language])
    if kernel_type:
        cmd.extend(["--kernel-type", kernel_type])
    if output_type:
        cmd.extend(["--output-type", output_type])
    if sort_by:
        cmd.extend(["--sort-by", sort_by])
    if csv:
        cmd.append("-v")
    return _run_as(cmd)


@mcp.tool()
def kaggle_kernels_list_files(
    kernel: str,
    csv: bool = False,
    page_size: Optional[int] = None,
    page_token: Optional[str] = None,
) -> str:
    """List kernel output files.

    Args:
        kernel: Kernel URL suffix in format <owner>/<kernel-name>.
        csv: Print results in CSV format.
        page_size: Items per page (max 200).
        page_token: Opaque page token for cursor-based paging.
    """
    cmd = ["kaggle", "kernels", "files", kernel]
    if csv:
        cmd.append("-v")
    if page_size is not None:
        cmd.extend(["--page-size", str(page_size)])
    if page_token:
        cmd.extend(["--page-token", page_token])
    return _run_as(cmd)


@mcp.tool()
def kaggle_kernels_push(
    path: Optional[str] = None,
    timeout: Optional[int] = None,
    accelerator: Optional[str] = None,
) -> str:
    """Push new code to a kernel (notebook/script) and run it.

    The folder must contain a valid kernel-metadata.json file.  See
    https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md

    Args:
        path: Folder for upload, containing data files and
            kernel-metadata.json (defaults to current working directory).
        timeout: Maximum run time in seconds.
        accelerator: Accelerator type (e.g. 'GPU', 'TPU').
    """
    cmd = ["kaggle", "kernels", "push"]
    if path:
        cmd.extend(["-p", path])
    if timeout is not None:
        cmd.extend(["-t", str(timeout)])
    if accelerator:
        cmd.extend(["--accelerator", accelerator])
    return _run_as(cmd, timeout=300)


@mcp.tool()
def kaggle_kernels_pull(
    kernel: str,
    path: Optional[str] = None,
    metadata: bool = False,
) -> str:
    """Pull down code from a kernel.

    Args:
        kernel: Kernel URL suffix in format <owner>/<kernel-name> or
            <owner>/<kernel-name>/<version>.
        path: Folder where files will be downloaded.
        metadata: Generate metadata file when pulling.
    """
    cmd = ["kaggle", "kernels", "pull", kernel]
    if path:
        cmd.extend(["-p", path])
    if metadata:
        cmd.append("-m")
    return _run_as(cmd)


@mcp.tool()
def kaggle_kernels_output(
    kernel: str,
    path: Optional[str] = None,
    force: bool = False,
    quiet: bool = False,
    file_pattern: Optional[str] = None,
) -> str:
    """Get data output from the latest kernel run.

    Args:
        kernel: Kernel URL suffix in format <owner>/<kernel-name>.
        path: Folder where files will be downloaded.
        force: Skip up-to-date check and force download.
        quiet: Suppress progress output.
        file_pattern: Regex pattern to match filenames. Only matching files
            are downloaded.
    """
    cmd = ["kaggle", "kernels", "output", kernel]
    if path:
        cmd.extend(["-p", path])
    if force:
        cmd.append("-o")
    if quiet:
        cmd.append("-q")
    if file_pattern:
        cmd.extend(["--file-pattern", file_pattern])
    return _run_as(cmd)


@mcp.tool()
def kaggle_kernels_status(kernel: str) -> str:
    """Display the status of the latest kernel run.

    Args:
        kernel: Kernel URL suffix in format <owner>/<kernel-name>.
    """
    return _run_as(["kaggle", "kernels", "status", kernel])


@mcp.tool()
def kaggle_kernels_delete(
    kernel: str,
    yes: bool = False,
) -> str:
    """Delete a kernel from Kaggle. Stops any running session.

    Use this to stop a kernel early mid-training. Destructive — removes all
    versions and outputs. If you need to preserve the kernel, use the Kaggle
    web UI's stop button instead.

    Args:
        kernel: Kernel URL suffix in format <owner>/<kernel-name>.
        yes: Skip confirmation prompt.
    """
    cmd = ["kaggle", "kernels", "delete", kernel]
    if yes:
        cmd.append("-y")
    return _run_as(cmd)


@mcp.tool()
def kaggle_kernels_logs(
    kernel: str,
    follow: bool = False,
    interval: Optional[int] = None,
) -> str:
    """Print execution logs from the latest kernel run.

    Args:
        kernel: Kernel URL suffix in format <owner>/<kernel-name>.
        follow: Continuously poll and print new log lines (like tail -f).
            NOTE: 'follow' will block until the kernel finishes or the
            timeout is reached.
        interval: Polling interval in seconds for follow mode (default 5).
    """
    cmd = ["kaggle", "kernels", "logs", kernel]
    if follow:
        cmd.append("-f")
    if interval is not None:
        cmd.extend(["--interval", str(interval)])
    return _run_as(cmd, timeout=600)


@mcp.tool()
def kaggle_kernels_logs_tail(
    kernel: str,
    lines: int = 30,
) -> str:
    """Fetch the last N lines of a kernel's execution log (non-blocking).

    Use this tool to poll a running kernel's health. Call it repeatedly
    (e.g. every 30–60 seconds) to monitor progress, check for errors, or
    verify that training is proceeding normally.

    Unlike `kaggle_kernels_logs(follow=True)`, this tool returns immediately
    and never blocks.

    Args:
        kernel: Kernel URL suffix in format <owner>/<kernel-name>.
        lines: Number of trailing log lines to return (default 30, max 200).
    """
    lines = max(1, min(lines, 200))
    raw = _run_as(["kaggle", "kernels", "logs", kernel], timeout=120)

    # If the command failed, return the error as-is.
    if raw.startswith("[exit code") or raw.startswith("[timeout") or raw.startswith("[error]"):
        return raw

    all_lines = raw.splitlines()
    if len(all_lines) <= lines:
        return raw

    truncated = all_lines[-lines:]
    header = f"--- showing last {lines} of {len(all_lines)} log lines ---"
    return header + "\n" + "\n".join(truncated)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@mcp.tool()
def kaggle_models_list(
    search: Optional[str] = None,
    sort_by: Optional[str] = None,
    owner: Optional[str] = None,
    page_size: Optional[int] = None,
    page_token: Optional[str] = None,
    csv: bool = False,
) -> str:
    """List Kaggle models.

    Args:
        search: Free-text search terms.
        sort_by: Sort order — 'hotness' (default), 'downloadCount',
            'voteCount', 'notebookCount', or 'createTime'.
        owner: Show models owned by a specific user or organization.
        page_size: Items per page (max 200).
        page_token: Opaque page token for cursor-based paging.
        csv: Print results in CSV format.
    """
    cmd = ["kaggle", "models", "list"]
    if search:
        cmd.extend(["-s", search])
    if sort_by:
        cmd.extend(["--sort-by", sort_by])
    if owner:
        cmd.extend(["--owner", owner])
    if page_size is not None:
        cmd.extend(["--page-size", str(page_size)])
    if page_token:
        cmd.extend(["--page-token", page_token])
    if csv:
        cmd.append("-v")
    return _run_as(cmd)


@mcp.tool()
def kaggle_models_get(
    model: str,
    path: Optional[str] = None,
) -> str:
    """Get a model's metadata.

    Args:
        model: Model URL suffix in format <owner>/<model-name>.
        path: Folder containing model-metadata.json.
    """
    cmd = ["kaggle", "models", "get", model]
    if path:
        cmd.extend(["-p", path])
    return _run_as(cmd)


@mcp.tool()
def kaggle_models_instances_list(
    model: str,
    csv: bool = False,
    page_size: Optional[int] = None,
    page_token: Optional[str] = None,
) -> str:
    """List model variations (instances) for a given model.

    Args:
        model: Model URL suffix in format <owner>/<model-name>.
        csv: Print results in CSV format.
        page_size: Items per page (max 200).
        page_token: Opaque page token for cursor-based paging.
    """
    cmd = ["kaggle", "models", "instances", "list", model]
    if csv:
        cmd.append("-v")
    if page_size is not None:
        cmd.extend(["--page-size", str(page_size)])
    if page_token:
        cmd.extend(["--page-token", page_token])
    return _run_as(cmd)


@mcp.tool()
def kaggle_models_instances_versions_list(
    model_instance: str,
    csv: bool = False,
    page_size: Optional[int] = None,
    page_token: Optional[str] = None,
) -> str:
    """List versions of a model variation.

    Args:
        model_instance: Model variation URL suffix in format
            <owner>/<model-name>/<framework>/<instance-slug>.
        csv: Print results in CSV format.
        page_size: Items per page (max 200).
        page_token: Opaque page token for cursor-based paging.
    """
    cmd = ["kaggle", "models", "instances", "versions", "list", model_instance]
    if csv:
        cmd.append("-v")
    if page_size is not None:
        cmd.extend(["--page-size", str(page_size)])
    if page_token:
        cmd.extend(["--page-token", page_token])
    return _run_as(cmd)


@mcp.tool()
def kaggle_models_instances_versions_download(
    model_instance_version: str,
    path: Optional[str] = None,
    untar: bool = False,
    force: bool = False,
    quiet: bool = False,
) -> str:
    """Download a model variation version.

    Args:
        model_instance_version: Version URL suffix in format
            <owner>/<model-name>/<framework>/<instance-slug>/<version>.
        path: Folder where files will be downloaded.
        untar: Untar the downloaded file (removes the tar).
        force: Skip up-to-date check and force download.
        quiet: Suppress progress output.
    """
    cmd = ["kaggle", "models", "instances", "versions", "download", model_instance_version]
    if path:
        cmd.extend(["-p", path])
    if untar:
        cmd.append("--untar")
    if force:
        cmd.append("-f")
    if quiet:
        cmd.append("-q")
    return _run_as(cmd)


# ---------------------------------------------------------------------------
# Config & Auth
# ---------------------------------------------------------------------------


@mcp.tool()
def kaggle_config_view() -> str:
    """View current Kaggle configuration values."""
    return _run_as(["kaggle", "config", "view"])


@mcp.tool()
def kaggle_config_set(name: str, value: str) -> str:
    """Set a Kaggle configuration value.

    Common keys: 'competition', 'path'.
    Args:
        name: Configuration key name.
        value: Configuration value.
    """
    return _run_as(["kaggle", "config", "set", name, value])


@mcp.tool()
def kaggle_config_unset(name: str) -> str:
    """Clear (unset) a Kaggle configuration value.

    Args:
        name: Configuration key name.
    """
    return _run_as(["kaggle", "config", "unset", name])


@mcp.tool()
def kaggle_quota(csv: bool = False) -> str:
    """Show the current user's weekly GPU and TPU accelerator quota.

    Args:
        csv: Print results in CSV format.
    """
    cmd = ["kaggle", "quota"]
    if csv:
        cmd.append("-v")
    return _run_as(cmd)


@mcp.tool()
def kaggle_auth_print_access_token() -> str:
    """Print an access token for the active Kaggle account."""
    return _run_as(["kaggle", "auth", "print-access-token"])


@mcp.tool()
def kaggle_auth_revoke(reason: Optional[str] = None) -> str:
    """Revoke the active profile's refresh token.

    After revocation the profile's token will no longer work.  Remove the
    profile with `kaggle_profile_remove` or re-add it with a fresh token.

    Args:
        reason: Optional reason for revoking the token.
    """
    cmd = ["kaggle", "auth", "revoke"]
    if reason:
        cmd.extend(["--reason", reason])
    return _run_as(cmd)
    return _run_as(cmd)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
