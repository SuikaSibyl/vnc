from __future__ import annotations

import json
from pathlib import Path


def _load_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _pair_baseline(method: str) -> str | None:
    if method == "video_nvdiffrast":
        return "nvdiffrast"
    if method == "video_pytorch3d":
        return "pytorch3d_sil_rgb"
    if method == "video_nvdiffrecmc":
        return "nvdiffrecmc"
    if method == "video_our_pytorch3d":
        return "our_pytorch3d"
    if method.startswith("prdpt_video_s"):
        return "prdpt_s" + method[len("prdpt_video_s") :]
    return None


def _build_summary(methods: dict[str, float]) -> dict:
    time_savings: dict[str, dict[str, float | str]] = {}
    for method, video_time_sec in sorted(methods.items()):
        baseline = _pair_baseline(method)
        if baseline is None or baseline not in methods:
            continue
        baseline_time_sec = methods[baseline]
        time_saved_sec = baseline_time_sec - video_time_sec
        time_savings[method] = {
            "baseline_method": baseline,
            "baseline_time_sec": baseline_time_sec,
            "video_time_sec": video_time_sec,
            "time_saved_sec": time_saved_sec,
            "time_saved_ratio": (time_saved_sec / baseline_time_sec) if baseline_time_sec else 0.0,
        }
    return {
        "methods": dict(sorted(methods.items())),
        "time_savings": time_savings,
    }


def write_method_json_timing_summary(out_dir: Path) -> Path:
    methods: dict[str, float] = {}
    for path in sorted(out_dir.glob("*.json")):
        if path.name == "optimization_time_summary.json":
            continue
        data = _load_json(path)
        if not data:
            continue
        method = data.get("method")
        total_time_sec = data.get("total_time_sec")
        if isinstance(method, str) and isinstance(total_time_sec, (int, float)):
            methods[method] = float(total_time_sec)

    summary_path = out_dir / "optimization_time_summary.json"
    summary_path.write_text(json.dumps(_build_summary(methods), indent=2))
    return summary_path


def write_subdir_summary_timing_summary(base_dir: Path) -> Path:
    methods: dict[str, float] = {}
    for path in sorted(base_dir.glob("*/summary.json")):
        data = _load_json(path)
        if not data:
            continue
        total_time_sec = data.get("total_time_sec")
        if isinstance(total_time_sec, (int, float)):
            methods[path.parent.name] = float(total_time_sec)

    summary_path = base_dir / "optimization_time_summary.json"
    summary_path.write_text(json.dumps(_build_summary(methods), indent=2))
    return summary_path


def write_explicit_timing_summary(base_dir: Path, methods: dict[str, float]) -> Path:
    summary_path = base_dir / "optimization_time_summary.json"
    summary_path.write_text(json.dumps(_build_summary(methods), indent=2))
    return summary_path
