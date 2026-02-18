"""Auto-detect CUDA/Python/Torch versions and install the matching flash-attn prebuild wheel."""

import importlib.metadata
import json
import platform
import re
import subprocess
import sys
import urllib.request

from rich.console import Console

console = Console()

REPO = "mjun0812/flash-attention-prebuild-wheels"
LATEST_API_URL = f"https://api.github.com/repos/{REPO}/releases/latest"


def get_python_tag() -> str:
    v = sys.version_info
    return f"cp{v.major}{v.minor}"


def get_cuda_version() -> str | None:
    try:
        import torch

        cuda_ver = torch.version.cuda
        if cuda_ver:
            return cuda_ver
    except (ImportError, AttributeError):
        pass

    try:
        out = subprocess.check_output(
            ["nvcc", "--version"], text=True, stderr=subprocess.DEVNULL
        )
        match = re.search(r"release (\d+\.\d+)", out)
        if match:
            return match.group(1)
    except FileNotFoundError:
        pass

    return None


def get_torch_version() -> str | None:
    try:
        import torch

        return torch.__version__.split("+")[0]
    except ImportError:
        return None


def fetch_available_wheels() -> tuple[str, list[str]]:
    """Fetch latest release tag and asset names from GitHub."""
    req = urllib.request.Request(
        LATEST_API_URL, headers={"Accept": "application/vnd.github+json"}
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())
    tag = data["tag_name"]
    wheels = [
        asset["name"]
        for asset in data["assets"]
        if asset["name"].endswith(".whl")
        and "linux_x86_64.whl" in asset["name"]
        and "manylinux" not in asset["name"]
    ]
    return tag, wheels


def parse_wheel_name(name: str) -> dict[str, str] | None:
    """Extract flash-attn version, cuda, torch, and python tags from a wheel filename."""
    m = re.match(
        r"flash_attn-([\d.]+)\+cu(\d+)torch([\d.]+)-(cp\d+)-(cp\d+)-linux_x86_64\.whl",
        name,
    )
    if not m:
        return None
    return {
        "fa_version": m.group(1),
        "cuda": m.group(2),
        "torch": m.group(3),
        "python": m.group(4),
        "filename": name,
    }


def cuda_ver_to_tag(cuda_ver: str) -> str:
    """Convert '12.8' → '128', '12.4' → '124'."""
    parts = cuda_ver.split(".")
    return parts[0] + parts[1]


def torch_ver_to_tag(torch_ver: str) -> str:
    """Convert '2.10.0' → '2.10'."""
    parts = torch_ver.split(".")
    return ".".join(parts[:2])


def _version_tuple(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in v.split("."))


def _pick_latest(candidates: list[dict[str, str]]) -> dict[str, str]:
    return max(candidates, key=lambda w: _version_tuple(w["fa_version"]))


def find_best_wheel(
    wheels: list[dict[str, str]], cuda_tag: str, torch_tag: str, py_tag: str
) -> dict[str, str] | None:
    matching = [w for w in wheels if w["torch"] == torch_tag and w["python"] == py_tag]
    if not matching:
        return None

    # exact cuda match — pick highest fa_version
    exact = [w for w in matching if w["cuda"] == cuda_tag]
    if exact:
        return _pick_latest(exact)

    # closest cuda (prefer highest <= user's)
    cuda_int = int(cuda_tag)
    lower = [w for w in matching if int(w["cuda"]) <= cuda_int]
    if lower:
        best_cuda = max(lower, key=lambda w: int(w["cuda"]))["cuda"]
        return _pick_latest([w for w in lower if w["cuda"] == best_cuda])

    # fallback: lowest available cuda
    lowest_cuda = min(matching, key=lambda w: int(w["cuda"]))["cuda"]
    return _pick_latest([w for w in matching if w["cuda"] == lowest_cuda])


def is_installed(package_name: str) -> bool:
    try:
        importlib.metadata.version(package_name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def ensure_flash_attn() -> bool:
    """Attempt to install flash-attn if not present. Returns True if installed/successful."""
    if is_installed("flash_attn"):
        return True

    if platform.system() != "Linux":
        # flash-attn prebuilds are Linux-only
        return False

    py_tag = get_python_tag()
    cuda_ver = get_cuda_version()
    if not cuda_ver:
        return False  # CUDA required

    torch_ver = get_torch_version()
    if not torch_ver:
        return False

    cuda_tag = cuda_ver_to_tag(cuda_ver)
    torch_tag = torch_ver_to_tag(torch_ver)

    console.print(
        f"[bold bright_cyan]⚡ Auto-installing flash-attn[/] "
        f"[dim](cu{cuda_tag} + torch{torch_tag} + {py_tag})[/]"
    )

    try:
        with console.status("[bold]Fetching wheels from GitHub…[/]"):
            release_tag, asset_names = fetch_available_wheels()

        parsed = [w for name in asset_names if (w := parse_wheel_name(name))]
        if not parsed:
            console.print("[red]✗ No wheels found in release.[/]")
            return False

        wheel = find_best_wheel(parsed, cuda_tag, torch_tag, py_tag)
        if not wheel:
            console.print("[red]✗ No matching wheel found.[/]")
            return False

        url = f"https://github.com/{REPO}/releases/download/{release_tag}/{wheel['filename']}"
        label = "exact" if wheel["cuda"] == cuda_tag else f"closest (cu{wheel['cuda']})"
        console.print(f"[green]✓[/] Found wheel ({label})")

        console.print("[dim]Installing flash-attn…[/]")
        subprocess.run(
            ["uv", "pip", "install", url, "--no-deps"],
            check=True,
            capture_output=True,
        )

        # Also ensure einops is installed as it's often needed with flash-attn models
        if not is_installed("einops"):
            console.print("[dim]Installing einops…[/]")
            subprocess.run(
                ["uv", "pip", "install", "einops"],
                check=True,
                capture_output=True,
            )

        console.print("[bold green]✓[/] flash-attn installed successfully!")
        return True

    except Exception as e:
        console.print(f"[red]✗ Auto-install failed: {e}[/]")
        return False
