"""Supported simulation-package boundary for the Radar 0.4 Studio adapters."""
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version


SUPPORTED_RUNTIME = {
    "witwin-radar": (0, 4),
    "witwin-channel": (0, 5),
}


def _major_minor(raw: str) -> tuple[int, int]:
    try:
        major, minor = raw.split(".", 2)[:2]
        return int(major), int(minor)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Unparseable simulation package version: {raw!r}") from exc


@lru_cache(maxsize=1)
def require_supported_runtime() -> dict[str, str]:
    """Refuse an absent or incompatible Radar/Channel line before a GPU solve."""
    observed = {}
    for distribution, required in SUPPORTED_RUNTIME.items():
        try:
            raw = version(distribution)
        except PackageNotFoundError as exc:
            raise RuntimeError(f"Required simulation package is not installed: {distribution}") from exc
        observed[distribution] = raw
        if _major_minor(raw) != required:
            expected = ".".join(str(value) for value in required)
            raise RuntimeError(
                f"{distribution} {raw} is unsupported; WT Radar requires the official {expected}.x release line."
            )
    return observed


__all__ = ["require_supported_runtime", "SUPPORTED_RUNTIME"]
